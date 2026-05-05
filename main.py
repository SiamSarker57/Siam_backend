import os
import io
import asyncio
import re
import csv
import json
import time
import random
import hashlib
import zipfile
import tarfile
import importlib
from datetime import datetime
from typing import Optional, List, Union, Any
from io import BytesIO

import joblib
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import optional dependencies
try:
    openpyxl = importlib.import_module("openpyxl")
    OPENPYXL_AVAILABLE = True
except ImportError:
    openpyxl = None
    OPENPYXL_AVAILABLE = False
    print("[WARNING] openpyxl not installed. Excel files will not be supported.")

try:
    pd = importlib.import_module("pandas")
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    print("[WARNING] pandas not installed. Some Excel features will be limited.")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("[WARNING] python-docx not installed. DOCX files will use basic extraction.")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("[WARNING] pyyaml not installed. YAML files will not be supported.")

# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SCALA-Guard API",
    description="Intelligent Open-Source Package Security Analysis Platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ML Model Load ───────────────────────────────────────────────────────────
# Load the trained malicious detection model
try:
    model = joblib.load('malicious_detection_model.pkl')
    MODEL_LOADED = True
    print("✓ Malicious detection model loaded successfully")
except Exception as e:
    model = None
    MODEL_LOADED = False
    print(f"[WARNING] ML model not found. Using simulated scoring. Error: {e}")

# Load feature columns
try:
    feature_cols = joblib.load('feature_columns.pkl')
    FEATURES_LOADED = True
    print(f"✓ Feature columns loaded: {len(feature_cols)} features")
except Exception:
    feature_cols = [
        'avg_established_duration', 'connections_per_second', 'failure_rate',
        'process_to_ip_diversity', 'destination_diversity_score', 'ip_entropy'
    ]
    FEATURES_LOADED = False
    print("[WARNING] Using default feature columns")

# Load scaler
try:
    scaler = joblib.load('scaler.pkl')
    SCALER_LOADED = True
    print("✓ Scaler loaded successfully")
except Exception:
    scaler = None
    SCALER_LOADED = False
    print("[WARNING] Scaler not found. Using raw features.")

# Load model metrics
try:
    model_metrics = joblib.load('model_metrics.pkl')
    print(f"✓ Model metrics loaded. Accuracy: {model_metrics.get('ensemble_accuracy', 0):.2%}")
except Exception:
    model_metrics = {}
    print("[WARNING] Model metrics not found")

# ─── In-Memory Scan History ───────────────────────────────────────────────────
scan_history: List[dict] = []

# ─── Pydantic Models ─────────────────────────────────────────────────────────
class PackageScanRequest(BaseModel):
    name: str
    version: Optional[str] = "latest"
    ecosystem: Optional[str] = "pypi"

class BatchScanRequest(BaseModel):
    packages: List[str]
    ecosystem: Optional[str] = "pypi"

class TextScanRequest(BaseModel):
    text: str
    ecosystem: Optional[str] = "pypi"

class PredictionRequest(BaseModel):
    values: List[float]

class AuthRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/login")
@app.post("/api/auth/login")
async def auth_login(req: AuthRequest):
    """Simple development authentication endpoint.
    Accepts email/password and returns a simple token for frontend testing.
    Replace with real auth (JWT, DB) for production.
    """
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    # Generate a simple deterministic token for demo purposes
    token_source = f"{req.email}:{time.time()}:{random.random()}"
    token = hashlib.sha256(token_source.encode()).hexdigest()

    return {"token": token, "email": req.email, "message": "Login simulated (dev mode)"}

# Database-backed user store
reset_tokens = {}  # token -> email
# store temporary OAuth states (dev only)
oauth_states = {}



class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class ForgotRequest(BaseModel):
    email: str

class ResetRequest(BaseModel):
    token: str
    new_password: str

@app.post('/auth/register')
@app.post('/api/auth/register')
async def register(req: RegisterRequest):
    if not req.email or not req.password or not req.name:
        raise HTTPException(status_code=400, detail='All fields are required')
    if req.email in users:
        raise HTTPException(status_code=400, detail='User already exists')
    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()
    users[req.email] = { 'name': req.name, 'password_hash': pwd_hash }
    token = hashlib.sha256(f"{req.email}:{time.time()}:{random.random()}".encode()).hexdigest()
    return { 'token': token, 'email': req.email, 'message': 'User registered (dev)'}

@app.post('/auth/forgot-password')
@app.post('/api/auth/forgot-password')
async def forgot_password(req: ForgotRequest):
    if not req.email:
        raise HTTPException(status_code=400, detail='Email required')
    # In real app send email. Here generate token and return it for dev
    token = hashlib.sha256(f"reset:{req.email}:{time.time()}:{random.random()}".encode()).hexdigest()[:16]
    reset_tokens[token] = req.email
    return { 'message': 'Reset token generated (dev). Use it to reset password.', 'reset_token': token }

@app.post('/auth/reset-password')
@app.post('/api/auth/reset-password')
async def reset_password(req: ResetRequest):
    if not req.token or not req.new_password:
        raise HTTPException(status_code=400, detail='Token and new_password are required')
    email = reset_tokens.get(req.token)
    if not email:
        raise HTTPException(status_code=400, detail='Invalid or expired token')
    pwd_hash = hashlib.sha256(req.new_password.encode()).hexdigest()
    users[email]['password_hash'] = pwd_hash
    # consume token
    del reset_tokens[req.token]
    return { 'message': 'Password has been reset' }

@app.get('/auth/google')
async def google_start():
    """Redirect user to Google's OAuth 2.0 consent screen."""
    client_id = os.getenv('GOOGLE_CLIENT_ID')
    if not client_id:
        # Helpful error instead of redirecting with missing client_id
        raise HTTPException(status_code=500, detail='Server OAuth misconfiguration: GOOGLE_CLIENT_ID is not set')

    redirect_uri = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8000/auth/google/callback')
    scope = 'openid email profile'
    state = hashlib.sha256(f"{random.random()}:{time.time()}".encode()).hexdigest()
    # store state in-memory (dev only)
    oauth_states[state] = True

    params = {
        'client_id': client_id,
        'response_type': 'code',
        'scope': scope,
        'redirect_uri': redirect_uri,
        'state': state,
        'access_type': 'offline',
        'prompt': 'consent'
    }

    # build URL safely
    from urllib.parse import urlencode
    query = urlencode(params)
    url = f'https://accounts.google.com/o/oauth2/v2/auth?{query}'
    return RedirectResponse(url)


@app.get('/auth/google/callback')
async def google_callback(code: str = None, state: str = None):
    """Handle OAuth callback from Google, exchange code for tokens, fetch profile."""
    if not code:
        raise HTTPException(status_code=400, detail='Missing code')

    client_id = os.getenv('GOOGLE_CLIENT_ID')
    client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
    redirect_uri = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8000/auth/google/callback')

    token_url = 'https://oauth2.googleapis.com/token'
    data = {
        'code': code,
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code'
    }

    token_resp = requests.post(token_url, data=data, headers={'Accept': 'application/json'})
    if token_resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f'Failed to exchange code: {token_resp.text}')
    token_json = token_resp.json()
    access_token = token_json.get('access_token')

    # Fetch user info
    userinfo_url = 'https://openidconnect.googleapis.com/v1/userinfo'
    ui_resp = requests.get(userinfo_url, headers={'Authorization': f'Bearer {access_token}'})
    if ui_resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f'Failed to fetch userinfo: {ui_resp.text}')
    profile = ui_resp.json()
    email = profile.get('email')
    name = profile.get('name') or email

    # create or find user
    if email not in users:
        users[email] = {'name': name, 'password_hash': ''}

    # issue app token
    app_token = hashlib.sha256(f"{email}:{time.time()}:{random.random()}".encode()).hexdigest()

    frontend = os.getenv('FRONTEND_URL', 'http://localhost:5173')
    # redirect back with token (use fragment to avoid leaking to servers)
    redirect_to = f"{frontend}/oauth-callback#token={app_token}&email={requests.utils.quote(email)}"
    return RedirectResponse(redirect_to)


@app.get('/auth/google/dev')
async def google_dev():
    # Development-only simulated Google sign-in (kept for compatibility)
    email = f"google_user_{int(time.time())}@example.com"
    name = 'Google User'
    token = hashlib.sha256(f"{email}:{time.time()}:{random.random()}".encode()).hexdigest()
    users[email] = { 'name': name, 'password_hash': '' }
    return { 'token': token, 'email': email }


# ─── KNOWN_MALICIOUS Set (keep your full list) ───────────────────────────────
KNOWN_MALICIOUS = {
    "10Cent10", "10Cent11", "11cent", "12cent", "1337c", "1337test", "1337z",
    "13cent", "14cent", "15cent", "16cent", "1inch", "2022-requests", "10Cent10", "10Cent11", "11cent", "12cent", "1337c", "1337test", "1337z",
    "13cent", "14cent", "15cent", "16cent", "1inch", "2022-requests", "282828282828282828",
    "3m-promo-link-gen", "3web", "3web-py", "4123", "90456984689490856", "a1rn", "AadhaarCrypt",
    "aaiohttp", "aasyncio", "acqusition", "activedevbadge", "adafruit_imageload", "adcv", "adhydra",
    "adm3", "adm4", "admcheck", "admcheck2", "adosint", "adpaypal", "adproof", "adpull", "adrandom",
    "adsplit", "adstr", "adurl", "adv2099m4", "adv2099m5", "adv2099m6", "adv2099m7", "advenced-requests",
    "advirtual", "advpruebitaa", "advpruebitaa3", "advpruebitaa4", "advpruebitaa6", "advpruebitaa8",
    "advpruebitaa9", "aeivasta", "aeodata", "aeodatav04", "aes44", "ahahjesus", "ai-solver-gen",
    "ai-solver-images", "ai-solver-py", "aidoc-consul", "aidoc-e2e-utils", "aietelegram", "aihottp",
    "Ailyboostbot", "Ailynitro", "ailzyn1tr0", "aio3", "aio5", "aioconsol", "aiogram-types", "aiohhttp",
    "aiohtpt", "aiohtt", "aiohttp-proxies", "aiohttp-proxy-connect", "aiohttp-socks4", "aiohttp-socks5",
    "aiohttp_proxies", "aiohttpp", "aioohttp", "aiosync", "aiothtp", "aiotoolsbox", "aiottp", "airduq",
    "airnitro", "algokit-arc", "alisdkcore", "alka10", "alzynitro", "amazonfdndn", "amazonpxnau",
    "aml-ds-pipeline-contrib", "amtplotlib", "analyze-me", "andex-maps", "android-plus-new", "androidspyeye",
    "ankpkg", "ankpkg1", "antchain-sdk-abcdjb", "antchain-sdk-abcdjb1", "antchain-sdk-acc", "antchain-sdk-account",
    "antchain-sdk-acm", "antchain-sdk-acs-iot", "antchain-sdk-baas-midway", "antchain-sdk-baasplus",
    "antchain-sdk-billing", "antchain-sdk-blockchain", "antchain-sdk-cafecmdb", "antchain-sdk-cas",
    "antchain-sdk-cat", "antchain-sdk-commercial", "antchain-sdk-commercialexternal", "antchain-sdk-das",
    "antchain-sdk-dog", "antchain-sdk-donpa", "antchain-sdk-ebc", "antchain-sdk-ent", "antchain-sdk-gatewayx",
    "antchain-sdk-goodschain", "antchain-sdk-iam", "antchain-sdk-industry", "antchain-sdk-loadtestmock",
    "antchain-sdk-mq", "antchain-sdk-ms", "antchain-sdk-mytc", "antchain-sdk-notification", "antchain-sdk-op",
    "antchain-sdk-pcc", "antchain-sdk-realperson", "antchain-sdk-rms", "antchain-sdk-shuziwuliu",
    "antchain-sdk-sp", "antchain-sdk-stlr", "antchain-sdk-tam", "antchain-sdk-tdm", "antchain-sdk-zjlm",
    "anticheatservice", "aohttp", "aowdjpawojd", "apch", "api-hypixel", "api-requester2", "apiclear",
    "apidev-coop", "apihypixel", "appetize-cli", "apple-py-music", "apple-tv-new", "aptx", "arangodb-driver",
    "arangodb-python", "arcalife", "archiveact", "args-python", "argspython", "argsreq", "artifact-lab-3-package-a18ff5d9",
    "artifact-lab-3-package-e90915e1", "aryi", "ascicolor", "ascii-chillah", "ascii2art", "ascii2text",
    "asn2crypto", "asptcer", "assuredserpentupload", "assyncio", "asteroid-filterbank", "astralnitro",
    "asyincio", "async-dispatcher", "asyncci", "asynccio", "asynci", "asyncii", "asynciio", "asyncio-box",
    "asyncio3", "asyncioi", "asyncioo", "asynciooo", "asynio", "asynncio", "asyyncio", "atplotlib",
    "auto_scrubber", "aws-login0tool", "awscl", "awsclie", "awsclii", "axderz", "axelo", "aysncio",
    "azerty123", "azure-cli-ml-preview", "azure-cli-ml-private-preview", "azure-sdk-tools",
    "azureml-contrib-jupyterrun", "azureml-contrib-optimization", "azureml-contrib-reports", "b2-sdk-python",
    "baeutifulsoup", "baeutifulsoup4", "bahaha", "bannerprint", "bannerscript", "barcodegeneratorqr",
    "barcodeqrgen", "baautifulsoup", "baautifulsoup4", "bbeautifulsoup", "bbitcoinlib", "bcrypto",
    "beaautifulsoup", "beaautifulsoup4", "BeaitifulSoop", "beaitifulsoup", "BeaotifulSoup", "beatifulsoup",
    "beatuifulsoup", "beatuifulsoup4", "BeaufifulSoup", "beauifulsoup", "beauitfulsoup", "beaurifulsoup",
    "beautfiulsoup", "beautiffulsoup", "beautiffulsoup4", "BeautifilSoop", "BeautifilSoup", "beautiflsoup",
    "beautiflsoup4", "beautiflulsoop", "beautiflulsoup", "beautiflusoup", "beautiflusoup4", "BeautifolSoup",
    "BeautifoulSoup", "beautifuklsoup", "beautifuksoup", "BeautifullSoop", "BeautifullSooup", "beautifullsoup",
    "beautifullsoup4", "beautifulosup", "beautifulosup4", "beautifuloup", "beautifuloup4", "BeautifulSoop",
    "beautifulsooup", "beautifulsooup4", "beautifulsop", "beautifulsop4", "beautifulsopu", "beautifulsou",
    "beautifulsou4", "BeautifulSoul", "beautifulsoup", "BeautifulSoup-new", "beautifulsoup-numpy",
    "beautifulsoup-requests", "beautifulsoup-scikit-learn", "beautifulsoup4-new", "beautifulsoup44",
    "beautifulsoupe", "BeautifulSoupo", "beautifulsoupp", "beautifulsoupp4", "beautifulsouup",
    "beautifulsouup4", "beautifulssoup", "beautifulssoup4", "beautifulsuop", "beautifulsuop4", "beautifulsup",
    "beautifulsup4", "beautifuosoup", "beautifusloup", "beautifusloup4", "beautifusoup", "beautifuulsoup",
    "beautifuulsoup4", "beautiifulsoup", "beautiifulsoup4", "beautilfulsoup", "beautiuflsoup", "beautiulsoup",
    "beautiulsoup4", "beauttifulsoup", "beauttifulsoup4", "BeautyfulSoup", "BeautySoup", "beauutifulsoup",
    "beauutifulsoup4", "bee23e3wddwwddwd23e2", "beeautifulsoup", "beeautifulsoup4", "beeee23323", "bestcolors",
    "bettercolor", "bettercolors", "betterstyle", "BeuatiflSoup", "beuatifulsoup", "beuatifulsoup4",
    "BeutifullSoup", "BeutifulSoop", "beutifulsoup", "beutifulsoup4", "bibp-utils", "bicoinlib", "bictoinlib",
    "bignum-devel", "biip-utils", "biitcoinlib", "bingchilling2", "binol", "bip-u8ls", "bip-uils", "bip-uitls",
    "bip-util", "bip-utilds", "bip-utile", "bip-utiles", "bip-utilos", "bip-utilss", "bip-utilz", "bip-utisl",
    "bip-utjls", "bip-utlils", "bip-uttils", "bip-uutils", "bipp-utils", "bips-utils", "bitccoinlib",
    "bitcinlib", "bitcionlib", "bitcoiinlib", "bitcoilib", "bitcoilnib", "bitcoinlb", "bitcoinlbi",
    "bitcoinli", "bitcoinlibb", "bitcoinliib", "bitcoinliv", "bitcoinnlib", "bitconilib", "bitconlib",
    "bitcooinlib", "bitocinlib", "bitoinlib", "bittcoinlib", "biup-utils", "blazeted", "bloxflipscraper",
    "bloxflipsearch", "blypack", "bobotosoji", "bogdi", "boogipoper", "boogishell", "boogishell2", "boogishell3",
    "boogishell4", "boogishell5", "boogishell6", "boostbot", "boostbot-api", "bootcampsystem", "booto3",
    "bop-utils", "botaa3", "boto33", "botoa", "botoa3", "botoo", "botoo3", "bpi-utils", "brokescolors",
    "brokescolors2", "brokescolors3", "brokesrcl", "browser-web", "browserdiv", "bruts", "bs3", "bs4tools",
    "bs5", "bsodinator", "btcoinlib", "bticoinlib", "btoocore", "builderknower", "builderknower2",
    "bup-utils", "bupi-utils", "burdoq", "business-kpi-manager", "bussardweg4a", "bussardweg4av2",
    "bussardweg4av3", "bytedtrace", "ca-certificates", "calculatingtime", "calculator",
    "calculator-2c397c49ab20c445", "calculator_2c397c49ab20c445", "calendar-extender", "camera-kit",
    "candyencode", "candyint", "candypong", "candyrandom", "capmoneercloudclient", "capmonsstercloudcliennt",
    "capmonsstercloudclient", "capmonster", "capmonster-task", "capmonsterccloudclient", "capmonstercloudclenet",
    "capmonstercloudclenit", "capmonstercloudclent", "capmonstercloudcliant", "capmonstercloudclieent",
    "capmonstercloudclieet", "capmonstercloudclien", "capmonstercloudcliend", "capmonstercloudcliendt",
    "capmonstercloudclienet", "capmonstercloudcliennt", "capmonstercloudclientt", "capmonstercloudcliet",
    "capmonstercloudcliient", "capmonstercloudclinent", "capmonstercloudclinet", "capmonstercloudclouidclient",
    "capmonstercloudcluodclient", "capmonsterclouddclient", "capmonsterclouddlient", "capmonsterclouidclient",
    "capmonsterclouudclient", "capmonstercludclient", "capmonstercoudclient", "capmonstercouldclient",
    "capmonsterrcloudclient", "capmosterclouclient", "capmostercloudclieent", "capmostercloudclienet",
    "capmostercloudclient", "capmostercloudclinet", "captcha-py", "captchaboy", "catbannerslol", "catbannersxd",
    "catme", "cccpu", "cccxt", "cchydra", "cckill", "cclgtb", "cclick", "ccmc", "ccolorama", "ccping",
    "ccryptocompare", "ccryptofeed", "ccsplit", "ccsv", "ccver", "ccx", "ccxtt", "ccxxt", "ceedee",
    "celery-poolroutes", "celery-routing", "celery-routr", "cephlib", "cert-orchestration-adapter",
    "certefi", "certif", "certife", "certifie", "certifiee", "cffii", "cffy", "chillie", "cick",
    "cikit-learn", "cilorama", "cipherbcrypt", "circleinfo", "cirtum", "cis-publishers", "ciscosparksdk",
    "citscapesscripts", "class-py", "clcik", "cleantalk", "clearapi", "clears", "clearsrc", "clicck",
    "clickk", "cliick", "clikc", "clistyling", "clolorama", "cloorama", "clorama", "cloroma", "cloud client",
    "cloud-client", "cloudfix", "clowpy", "code-helper", "codeui", "codigosintaxis", "coingecko-apis",
    "coinmarketcaps", "colarama-api", "colarg", "colargs", "colarify", "colarise", "colaroma", "colimer",
    "collorama", "collored", "coloama", "coloarma", "colomara", "coloorama", "colopym2", "color-fade",
    "color-utility", "color-vividpy", "coloraa", "colorahma", "coloram", "colorama-py", "coloramaa",
    "coloramae", "coloramah", "coloramal", "coloramas", "coloramaz", "colorame", "coloramia", "coloramka",
    "coloramma", "coloramna", "coloramo", "coloramoo", "coloramqa", "coloramqs", "coloramu", "coloramwa",
    "coloramws", "coloramxa", "coloramxs", "coloramza", "coloramzs", "colorana", "coloranam", "colorara",
    "coloratem", "colorating", "colorayma", "colored-fidget", "colored-upgrade", "colorema", "coloreq",
    "coloreq1", "coloreq2", "coloreq3", "colorfade", "colorfades", "colorfidget", "colorhrama", "coloriv",
    "colorized", "colorizng", "colorm", "colormma", "coloroama", "coloroma", "colorram", "colorrama",
    "colorramma", "colors-it", "colors-update", "colors5", "colorsama", "colorsapi", "colorscmd",
    "colorsgradient", "colorslib", "colorslog", "colorstyle", "colorwed", "colorwin", "colorzing",
    "colouorama", "colourama", "colouramas", "colourfool", "colprama", "colurama", "compilecls", "conda-verifyyyyy",
    "conio", "consolecmds", "controlinfo", "controlmine", "controlpong", "controlre", "controlreplace",
    "controlvirtual", "controlvisa", "cookiesjar", "coolrama", "coolstyle", "coorama", "cordipy", "corlorama",
    "cosmostroposphere", "costrar", "coulor", "cpuhydra", "cpukill", "cpuping", "cpuproof", "cpupy", "cpuver",
    "cra-template-snap", "craftcpu", "craftencode", "craftmc", "craftosint", "craftproof", "craftrand",
    "craftreplace", "crptocompare", "crptofeed", "crpytocompare", "crpytofeed", "crpytography", "crryptocompare",
    "crryptofeed", "crypocompare", "crypofeed", "crypotcompare", "crypotfeed", "crypptocompare", "crypptofeed",
    "cryptcompare", "cryptcoompare", "cryptfeed", "cryptfoeed", "crypto-data-fetch", "crypto-get-price",
    "crypto-open", "crypto-os", "crypto-pygame", "cryptobalance", "cryptoccompare", "cryptocmopare",
    "cryptocmpare", "cryptocomapre", "cryptocomare", "cryptocommpare", "cryptocompaare", "cryptocompae",
    "cryptocompaer", "cryptocompar", "cryptocomparee", "cryptocomparre", "cryptocomppare", "cryptocomprae",
    "cryptocompre", "cryptocoompare", "cryptocopare", "cryptocopmare", "cryptoeed", "cryptoefed", "cryptofed",
    "cryptofede", "cryptofee", "cryptofeedd", "cryptofeeed", "cryptoffeed", "cryptographylib", "cryptographylibary",
    "cryptographylibs", "cryptographyy", "cryptographz", "cryptograpyh", "cryptolibs", "cryptoocmpare",
    "cryptoocompare", "cryptoofeed", "cryptoompare", "crypttocompare", "crypttofeed", "Crystalnitro",
    "crytic_compilers", "crytocompare", "crytofeed", "crytpocompare", "crytpofeed", "cryyptocompare",
    "cryyptofeed", "Cryztalnitro", "csikit-learn", "csrapy", "cstlsy", "cstmotkinter", "csvv", "ctfpipshell",
    "ctx", "ctyps", "cuatomtkinter", "culturestreak", "cuolor", "cuolur", "cusgtomtkinter", "cusromtkinter",
    "custm", "custmtkinter", "custmtokinter", "custogtkinter", "custohtkinter", "custojmtkinter", "custojtkinter",
    "custoktkinter", "customekinter", "customkinter", "customtikinter", "customtiknter", "customtinter",
    "customtjinter", "customtkfnter", "customtkibter", "customtkihter", "customtkimter", "customtkinber",
    "customtkinet", "customtkinetr", "customtkinger", "customtkingter", "customtkinrer", "customtkintar",
    "customtkinte", "customtkinted", "customtkinteer", "customtkintert", "customtkintet", "customtkintre",
    "customtkintrer", "customtkintrr", "customtkintwr", "customtkinyer", "customtkitenr", "customtkiter",
    "customtkitner", "customtkitnerr", "customtkitnre", "customtkiyter", "customtkjnter", "customtkknter",
    "customtkniter", "customtkniterr", "customtknster", "customtknter", "customtkwnter", "customtkznter",
    "custontkinter", "custoqtkinter", "custotinter", "custotkinter", "custotkminter", "custotminter",
    "custoumtkinter", "custpmtkinter", "custrmtkinter", "custumtkinter", "custvomtkinter", "cutomtkinter",
    "cuwtomtkinter", "cuxtomtkinter", "cvad", "cvhttp", "cvint", "cvmc", "cvnvidia", "cvosint", "cvpaypal",
    "cvping", "cvram", "cvstudy", "cvultra", "cvver", "cxct", "cxt", "cyphers", "cypress", "cyptocompare",
    "cyptofeed", "cyrptocompare", "cyrptofeed", "darkmanontop", "dasdhaiusd1212", "dasdsajdjsaasddsad",
    "dasdsajdjsadsad", "data-platform-airflow-operators", "data-platform-airflow-recipes", "data-platform-dbt",
    "data-platform-observability-core", "data-platform-observability-validation", "dataclasses-python-version",
    "datagraph", "dbcounter", "dccsx", "dcreq", "ddiscord-py", "ddiscord-webhook", "deeepl", "deep-translate",
    "deep-translation", "deepmountains-lrce", "Deeprce", "defca", "deflib", "dekugay", "demo-malicious-package",
    "demontonto", "demontonto1", "demopaxkhimkus", "dependency-conf", "dependency1338", "dependency999",
    "dependency_confusion123", "dependency_confusion123456", "dependencyrrr", "dequests", "derkpy",
    "detection-telegram", "detection_telegram", "devicespoof", "devicespoofer", "devide04", "dfdfdfdfhhh",
    "dhamaka", "dhawnz", "diaossama-test1", "dicord-py", "dicord-webhook", "dicsord-py", "dicsord-webhook",
    "dicuser", "dider", "diiscord-py", "diiscord-webhook", "directui", "disccord-py", "disccord-webhook",
    "discmusic", "discod-py", "discod-webhook", "discodr-py", "discodr-webhook", "discomusic", "discoord-py",
    "discoord-webhook", "discor-webhook", "discord-addon", "discord-commands", "discord-dev", "discord-ebhook",
    "discord-ewbhook", "discord-gift", "discord-hook", "discord-manager", "discord-p", "discord-ppy",
    "discord-py-bot", "discord-pyy", "discord-simple-http", "discord-slash", "discord-solver", "discord-wbehook",
    "discord-wbhook", "discord-webbhook", "discord-webhhook", "discord-webhok", "discord-webhoko",
    "discord-webhoo", "discord-webhookk", "discord-webhoook", "discord-webohok", "discord-webook",
    "discord-weebhook", "discord-wehbook", "discord-wehook", "discord-wwebhook", "discord-y", "discord-yp",
    "discord.pt", "discord.pu", "discordbotapi", "discordcmd", "discordcolor", "discordd-py", "discordd-webhook",
    "discorder", "discordhook", "discordreq1", "discordreq99", "discordreqq", "discordrequ", "discordrolex",
    "DiscordSafety", "discordwebutil", "discordwebutils", "discorrd-py", "discorrd-webhook", "discotools",
    "discrd-py", "discrd-webhook", "discrod-py", "discrod-webhook", "disocrd-py", "disocrd-webhook",
    "disord-py", "disord-webhook", "disscord-py", "disscord-webhook", "distrib", "disutil", "diuser",
    "dividedinkedwarpdrive", "djanga", "djanggo", "django-log-tracker", "django-metamaks-auth",
    "django-metamask-aut", "django-pyyaml", "django-web2-auth", "django-web3-aut", "django-web4-auth",
    "djangoo", "DoNotDownload", "dont-test-me", "dotencode", "douctils", "downrun", "dpp-client",
    "dpp-client1234", "dpp_client", "dpp_client1234", "dpy-bot", "drgn-tokenization", "driftme",
    "dscord-py", "dscord-webhook", "dscripting", "dshttpslib", "dsicord-py", "dsicord-webhook", "dumers",
    "duonet", "easycap", "easyfuncsys", "easygetflag", "easyhttprequest", "easyinstall", "easyrequests",
    "easytimestamp", "eautifulsoup", "eautifulsoup4", "ebautifulsoup", "ebautifulsoup4", "ebsocket-client",
    "ebsockets", "edcv", "edenred", "edgame", "edgehttp", "edgui", "edhttp", "edinfo", "edintel", "edosint",
    "edpaypal", "edpost", "edpull", "edram", "edre", "edreplace", "edurl", "edvisa", "eeAjHjmCLAkF",
    "eeeeeeeeeeeee344324f", "eepl", "eethereum", "eetherium", "ef3233434refefeffe", "ef323refefeffe",
    "ef334343rf3feefefefefeffeefeffe", "effre4frferfrf", "ehtereum", "elenium", "emoji-country-flags", "emoki",
    "enchantv", "encodecontrol", "encodecpu", "encodelgtb", "encodeload", "encodemask", "encodenvidia",
    "encodeping", "encodepong", "encodetool", "encodever", "encodevisa", "ensorflow", "enumerate-iam-aws",
    "EoerbIsjxqyV", "equests", "equests-toolbelt", "erquests", "erquests-toolbelt", "eslenium",
    "esqadcpuhacked", "esqadgameultra", "esqadhackedosint", "esqadinfo", "esqadintellgtb", "esqadintelpong",
    "esqadrandom", "esqcandygrandtool", "esqcandyinfo", "esqcandyosint", "esqcandyosintcandy",
    "esqcandysplitpep", "esqccinfo", "esqccmask", "esqccpongcpu", "esqccpullpush", "esqccpy", "esqccre",
    "esqccreplaceintel", "esqccstringmask", "esqccstudyhacked", "esqccvirtual", "esqccvisa",
    "esqcontrolgrandsplit", "esqcontrolhttphttp", "esqcontrolint", "esqcontrollibmine", "esqcpugrandpong",
    "esqcpuguilgtb", "esqcpupipkill", "esqcpupy", "esqcpurand", "esqcpustudyhydra", "esqcrafted",
    "esqcraftguimine", "esqcraftlibcv", "esqcraftremc", "esqcraftverhacked", "esqcraftvirtual",
    "esqcraftvisaproof", "esqcvcpupy", "esqcved", "esqcvinfo", "esqcvinfoed", "esqcvinfogrand",
    "esqcvltgbhydra", "esqcvpost", "esqcvproof", "esqcvultralgtb", "esqcvvermask", "esqedencodeurl",
    "esqedpushvm", "esqedsplit", "esqencodecvpush", "esqencodegameproof", "esqencodemask", "esqencodemc",
    "esqencodepaypalpong", "esqencodepostinfo", "esqencodereplaceproof", "esqgamecandy", "esqgameencodereplace",
    "esqgameguiintel", "esqgameinfo", "esqgameloadgui", "esqgameloadrandom", "esqgamemask", "esqgamemineping",
    "esqgameram", "esqgamerandpull", "esqgamereplacetool", "esqgamestringvm", "esqgetcontrolpost",
    "esqgetinfoping", "esqgetintel", "esqgetlibpyw", "esqgettool", "esqgettoolvm", "esqgetultragui",
    "esqgeturl", "esqgrandcandy", "esqgrandcandyproof", "esqgrandminetool", "esqgrandpaypal", "esqgrandpull",
    "esqgrandpyinfo", "esqgrandsuper", "esqgrandvirtualget", "esqgrandvm", "esqguiload", "esqguipippost",
    "esqguipong", "esqguipostpost", "esqguiproofad", "esqguistudy", "esqhackedlgtbpip", "esqhackedmask",
    "esqhackednvidiamine", "esqhackedrandomcc", "esqhttpguicc", "esqhttpinfo", "esqhttplgtb",
    "esqhttpmaskload", "esqhttppaypalurl", "esqhttppywinfo", "esqhttpreplace", "esqhttpvmurl",
    "esqhydraedcc", "esqhydraload", "esqhydraproof", "esqhydraramhydra", "esqhydrastudyre", "esqhydrasuper",
    "esqhydratool", "esqinfohackednvidia", "esqinfohttppush", "esqinfointproof", "esqinfoloadnvidia",
    "esqinforandom", "esqinfostr", "esqinfostrget", "esqinfostrosint", "esqinfovirtualgame", "esqintcvgui",
    "esqintelcandyload", "esqintelgrand", "esqintelinfo", "esqintelosint", "esqintelping", "esqintelpush",
    "esqintelram", "esqintelreplace", "esqintmaskgui", "esqintpaypalram", "esqintrandhttp", "esqintreplace",
    "esqintstringcraft", "esqintstudyhacked", "esqkillgame", "esqkillpongrand", "esqkillproofcv",
    "esqkillpywreplace", "esqkillsplitpull", "esqkillstring", "esqkillultrarand", "esqkillurlcraft",
    "esqlgtbad", "esqlibadpong", "esqlibcontrol", "esqlibcpucc", "esqlibcpuosint", "esqlibkill",
    "esqlibkillstr", "esqlibpiptool", "esqlibpost", "esqlibpullreplace", "esqlibstudy", "esqlibtoolhttp",
    "esqlibver", "esqloadcandy", "esqloadcved", "esqloadhackednvidia", "esqloadvisa", "esqmaskcontrolgrand",
    "esqmaskcpuhacked", "esqmaskcpustring", "esqmaskcraftnvidia", "esqmaskgrandcc", "esqmaskintlib",
    "esqmaskintnvidia", "esqmaskpepget", "esqmaskpepgui", "esqmasksplitpush", "esqmaskstr", "esqmaskvirtualcpu",
    "esqmced", "esqmcget", "esqmcreplacestr", "esqminead", "esqminecv", "esqmineencodekill", "esqminegrandget",
    "esqminekill", "esqminelib", "esqmineosintpip", "esqminepymc", "esqminepysuper", "esqminestrpong",
    "esqnvidiaadram", "esqnvidiakill", "esqnvidiamaskvirtual", "esqnvidiapywpip", "esqnvidiasplit",
    "esqnvidiastringcv", "esqnvidiaultrapep", "esqnvidiavmget", "esqosintcandycc", "esqosintcraftram",
    "esqosinthackedstudy", "esqosintminehydra", "esqosintnvidia", "esqosintpull", "esqosintvisaget",
    "esqpaypalcraft", "esqpaypalgamemc", "esqpaypalhydra", "esqpaypalinfogui", "esqpaypalnvidiaurl",
    "esqpaypalpongcc", "esqpaypalpulled", "esqpaypalvirtualmc", "esqpaypalvisaencode", "esqpephydrapaypal",
    "esqpepintpyw", "esqpeppost", "esqpeprandpaypal", "esqpepstudy", "esqpingcpu", "esqpingencode",
    "esqpinghackedvisa", "esqpingint", "esqpingpingcontrol", "esqpingpost", "esqpingurl", "esqpingvm",
    "esqpippip", "esqpostcpu", "esqpostcraft", "esqposthackedver", "esqpostintel", "esqpostintelcpu",
    "esqpostloadping", "esqpostrandomstring", "esqproofinthydra", "esqproofosintkill", "esqproofpongint",
    "esqproofpostpull", "esqproofpostvisa", "esqproofproof", "esqproofpushlib", "esqproofrandommine",
    "esqproofurlstudy", "esqproofvisacandy", "esqpullcontrolint", "esqpullhackedstudy", "esqpullpep",
    "esqpullpull", "esqpushadcpu", "esqpushhttp", "esqpushlgtb", "esqpushmcstring", "esqpushrehttp",
    "esqpushstudy", "esqpycraftpost", "esqpyhttp", "esqpyhydrarandom", "esqpyinted", "esqpymask",
    "esqpymcpy", "esqpysuperinfo", "esqpywcv", "esqpywencode", "esqpywlgtbnvidia", "esqpywlibencode",
    "esqpywload", "esqpywpepcraft", "esqpywramint", "esqpywrandomstr", "esqramhttp", "esqramhydra",
    "esqramreplace", "esqrandadget", "esqrandmask", "esqrandomgui", "esqrandominfo", "esqrandomintel",
    "esqrandompullad", "esqrandompullmine", "esqrandompushgrand", "esqrandomreplacecc", "esqrandpippush",
    "esqrandpy", "esqrandram", "esqrandultra", "esqrandver", "esqrecpuvm", "esqrecv", "esqreinfore",
    "esqrenvidiaad", "esqreplacegrandpep", "esqreplacegrandpyw", "esqreplaceguistudy", "esqrepullstring",
    "esqrepy", "esqrepymc", "esqrere", "esqrerecandy", "esqrestrhttp", "esqresuper", "esqreultramask",
    "esqrevmgrand", "esqsplitmaskpy", "esqsplitosintint", "esqsplitpepcandy", "esqsplitpepcv",
    "esqsplitpeptool", "esqsplitping", "esqsplitpushpush", "esqsplitpy", "esqsplitsplitver", "esqsplitstr",
    "esqstringgamere", "esqstringminecv", "esqstringnvidia", "esqstringpaypal", "esqstringpepcv",
    "esqstringpingcraft", "esqstringpong", "esqstringramgui", "esqstringsupergui", "esqstringtoolhacked",
    "esqstringtoolint", "esqstringvirtual", "esqstringvirtualtool", "esqstrintelsplit", "esqstrosint",
    "esqstrping", "esqstrpushgrand", "esqstrpushsplit", "esqstrstudy", "esqstudycpu", "esqstudyedgrand",
    "esqstudyint", "esqstudyinttool", "esqstudypiprand", "esqstudypongcandy", "esqstudyrepull", "esqsupergame",
    "esqsuperosintpost", "esqsuperpyw", "esqtoolinfoultra", "esqtoolnvidia", "esqtoolpoststr", "esqtoolrand",
    "esqtoolstr", "esqtoolverver", "esqtoolvisaram", "esqultracc", "esqultraLGTBhydra", "esqultraosintlgtb",
    "esqultrareplacecandy", "esqultraultrapong", "esqultravirtualstring", "esqurlencodead", "esqurlget",
    "esqurlintel", "esqurlosinthydra", "esqveradvirtual", "esqverccpush", "esqvergamepy", "esqverinfo",
    "esqverlibosint", "esqverosintint", "esqverpostcontrol", "esqverpyosint", "esqvirtualedhacked",
    "esqvirtualgame", "esqvirtualintpep", "esqvirtualkillint", "esqvirtualkillsuper", "esqvirtualpush",
    "esqvisaencode", "esqvisakill", "esqvisakilled", "esqvisamaskpyw", "esqvisapullre", "esqvisapyw",
    "esqvisastred", "esqvisaurlstring", "esqvisaverload", "esqvisavirtualencode", "esqvmgetram",
    "esqvmmineping", "esqvmpep", "esqvmverpep", "estimating", "etehreum", "eth-keccak", "eth-manager",
    "etheereium", "etheerem", "etheereum", "etheerim", "etheerium", "etheeruim", "etheeruimm", "etheerum",
    "etheeruum", "etheirum", "etheraem", "etherapi", "ethereim", "ethereium", "etherem", "ethererum",
    "ethereuim", "ethereum2", "ethereumm", "ethereun", "ethereuum", "etheriem", "etherim", "etheriuim",
    "etherium", "etheriumm", "etheriun", "etheriuum", "etherreeum", "etherreum", "etherreumm", "etherrium",
    "etherriuum", "etherriuumm", "ethertoollerz", "etheruem", "etheruemm", "etheruim", "etherum", "etherumm",
    "etherun", "etheruum", "etheum", "etheurm", "ethherium", "ethherum", "ethreeum", "ethreium", "ethreum",
    "ethrum", "ethter", "ethtoolz", "etlisalat", "etnsorflow", "ettherium", "etuptool", "etuptools",
    "eutherium", "EvannLeGoat", "evilshield", "ewb3-py", "ewbsocket-client", "excaliburx", "exporter-stackdriver",
    "extracolorsv2", "ezbeamer", "ezbeamsw", "faiss-gp", "FakePipD", "fakessh", "falsk", "fantastic-ascii",
    "faq", "farking", "fasdghjkhjafsd", "fast-httpx", "fast_httpx", "fastapi-https", "fastapi-toolkit",
    "fastapi_toolkit", "fastpep8", "fatnoob", "fbdebug", "fdkit", "fdsfsdfsdfgsdg", "feenton", "fef3434334dwrg",
    "fefeefrrg", "felpesviadinho", "feqtrade", "fequests", "ferned", "ferns", "ferqtrade", "feur",
    "ffreqtrade", "figflix", "figlets", "fijiwashere12323", "firefoxupdate", "firstbasicpyapp", "flagui",
    "flak7", "flak8", "flake7", "flask-requests-complex", "flexponlib", "flsak", "fluiddaddy", "fnbot2",
    "fncache", "forenitq", "forenits", "forenity", "forenitz", "forestyle", "ForgePy", "ForgePys", "ForgyP",
    "ForgyPs", "forings", "forring", "fortynite", "foxx-py", "fpsboost", "fredli", "free-net-vpn",
    "free-net-vpn2", "free-requests-module", "freeqtrade", "frefereffee", "frekpy", "freqqtrade", "freqrade",
    "freqrtade", "freqtade", "freqtarde", "freqtraade", "freqtrad", "freqtradde", "freqtradee", "freqtrae",
    "freqtraed", "freqtrdae", "freqtrde", "freqtrrade", "freqttrade", "fretqrade", "fretrade", "friendlyproxies",
    "frqetrade", "frqtrade", "frreqtrade", "fstcall", "fuzywuzy", "fuzywuzzy", "fuzzywuzy", "fuzzzywuzzy",
    "fxzontop", "fyinance", "gameintel", "gamepass", "gameproof", "gamescodes", "gamesplit", "gamestr",
    "gamestring", "gasmanez", "gen-agent-fingerprint", "generaldelta", "generator-meeseeks", "genesisbot",
    "genui", "gequests", "gesnim", "getcandy", "getcc", "getcontrol", "getgui", "getkill", "getlatency",
    "getmine", "getping", "getpost", "getsuper", "getwebshare", "getwebshareproxy", "ggitpython", "ggvpslmao",
    "giitpython", "giptython", "gipython", "gitppython", "gitpthon", "gitptyhon", "gitpyhon", "gitpyhton",
    "gitpythhon", "gitpythn", "gitpythno", "gitpythonn", "gitpythoon", "gitpytohn", "gitpyton", "gitpytthon",
    "gitpyython", "gittpython", "gitypthon", "gitython", "gkjzjh146", "glovo-data-platform-declarative",
    "glovo-data-platform-declarative-airflow", "glovo-data-platform-importer-brain", "gmgeoip", "go-requests",
    "goldensweatshirtwifi", "good-regex", "google-requests", "gradientcolors", "grandcc", "grandhydra",
    "grandint", "grandkill", "grandmask", "grandmc", "grandmine", "grandslam", "grandstr", "grandver",
    "graphcore-cloud-tools", "graphene-arangodb", "gtipython", "gtpython", "gui-build", "guicv", "guied",
    "guihacked", "guihydra", "guikill", "guinvidia", "guipep", "guiproof", "guipush", "guipyw", "guirandom",
    "guisplit", "guistr", "gunbase", "guypy", "h8shdf89d", "habboapps", "habibisus", "hackedcc", "hackedcraft",
    "hackedget", "hackedhydra", "hackedintel", "hackedload", "hackedmine", "hackednvidia", "hackedping",
    "hackedpost", "hackedpush", "hackedstudy", "hackedtool", "hackedvisa", "hackerfilelol", "hackerfileloll",
    "haisenbergs-pkg", "HakePip", "hameni", "hansont", "hase", "hashdecrypt", "hashdecrypts", "haxorsiambot",
    "hazard", "hello-world-exampl", "hello-world-example", "hellomynameisahjahs", "hellowhatisao", "helmerter",
    "helrmerter", "henter", "hexcolured", "hexmanibm", "hexteamibm", "historic-crypt", "hkg-sol-utils",
    "hnUHfYZUmKMO", "hoes", "hreading", "hsbcgui", "httiop", "http-interact", "http3-client", "http3_client",
    "httpacc", "httphacked", "httphydra", "httplat", "httplgtb", "httpnvidia", "httpproof", "httpre",
    "httpreplace", "httprequesthub", "HTTPRequesting", "httpscolor", "httpsing", "httpslib", "httpsp",
    "httpsreqfast", "httpsrequestsfast", "httpssp", "httpssus", "httpsus", "httptes232", "httpuri", "httpvisa",
    "httpx-advanced2", "httpx-advanced3", "httpxboost", "httpxc", "httpxfaster", "httpxgetter", "httpxmodifier",
    "httpxontop", "httpxrequester", "httpxrequesterv2", "httpxs", "huehuehuehue", "hugebbc", "huluquests",
    "humanqueen", "humanqueenn", "hydragame", "hydraget", "hydragui", "hydrahttp", "hydrahydra", "hydrainfo",
    "hydraintel", "hydralib", "hydraload", "hydramask", "hydramine", "hydrapaypal", "hydrapep", "hydraping",
    "hydraproof", "hydrapyw", "hydrarand", "hydrastr", "hydrastudy", "hydratool", "hydraurl", "hydravirtual",
    "hydravisa", "hymcapi", "hypixel-coins", "hypixel-networth-api", "hypixelmc", "iam-enumerate", "iaohttp",
    "ibtcoinlib", "icedgen", "idscord-py", "idscord-webhook", "idtotoken", "igtool", "igtoolz", "igtpython",
    "iiris-new", "imagesolverpy", "imapping", "implejson", "important-package", "importantpackage",
    "importlib-metadate", "importscraper", "inbm-lib", "IncapError", "incrivelsim", "inda", "infoglmi",
    "infohydra", "infoint", "infonvidia", "infoosint", "infopaypal", "infoping", "infosys", "initializers",
    "inject-rem", "insanepackage217234234242423442983", "insanepackage21724342386744243242983",
    "insanepackage217424422342983", "insanepackage2179824234242342433", "insanepackageongong11192",
    "insanepackageongong192", "insanepackagev1414", "instabots", "install-crypto", "install-pytest",
    "install-pyyaml", "installpippython", "installpy", "intcontrol", "intelcraft", "intelgrand", "intellib",
    "intelpong", "intelproof", "intelpush", "intelpy", "intelrand", "intelstr", "intelvisa", "intget",
    "inthacked", "intint", "intlib", "intmc", "intnvidia", "intpaypal", "intpep", "intpyw", "intstudy",
    "intver", "ipaddres", "ipadress", "ipboards", "ipg", "ipllow", "ipyhton", "ironic-secureboot-driver",
    "iscord-py", "iscord-webhook", "ismplejson", "itcoinlib", "itpython", "j3y5r", "ja3-hashscript",
    "jango-metamask-auth", "jango-web3-auth", "jas9do1", "javapatch", "jeilyfish", "jfqyotpnvb", "jksandbox",
    "jmdrs", "johnhammondfanpackage124", "johnhammondontop183", "judyb-advanced", "juk", "junkeldat",
    "jupitercalc", "jupyter-calendar-extension", "jupyter-pytest-fi-console", "jupyter_calendar_extension",
    "jxhcfdwvun", "jxzspwqwup", "jzyRLjROXlCa", "kangpy", "kayauthgen", "kazer12", "kcalendar", "kears",
    "keras-arg", "keras-beautifulsoup", "kers", "keyauthkey", "keybaord", "kfactionantiafk", "kfactionbypasser",
    "kfactionlogger", "kfactionlogger2", "killcc", "killcontrol", "killcraft", "killed", "killgame",
    "killhacked", "killhydra", "killosint", "killproof", "killpush", "killpyw", "killrand", "killreplace",
    "killskids-auth", "killvisa", "kings-landing-obfuscate", "kjfgjsfgj", "kjfgjsfgj1", "kjfgjsfgj12",
    "kjfgjsfgj123", "ktcalendar", "ktusvgeqiu", "kubespy", "kynlocker", "l7", "lalaproxy", "lastweb3toolzbro",
    "laysound", "lcick", "ld-impl-linux", "ld-impl-linux-64", "ld_impl_linux-64", "lgtbad", "lgtbcraft",
    "lgtbhttp", "lgtbpep", "lgtbpost", "lgtbpull", "lgtbre", "lgtbsplit", "lgtbstr", "lgtbultra", "lgtbvm",
    "libadcvcpu", "libadstringstudy", "libadtool", "libadtoolmc", "libadtoolrandom", "libaryscraper",
    "libblas3", "libcandyadhydra", "libcandycraftcontrol", "libcandyedpong", "libcandygame",
    "libcandyintelvisa", "libcandykillosint", "libcandyosint", "libcandyreplacecandy", "libcandystr",
    "libccint", "libcckillhydra", "libccreplacemask", "libcontrol", "libcontrolgui", "libcontrolguipyw",
    "libcontrolhttpstr", "libcontrolinfostr", "libcontrolminepyw", "libcontrolosint", "libcontrolrehydra",
    "libcontrolstringcc", "libcontroltoolver", "libcontrolultravm", "libcontrolurl", "libcontrolverlgtb",
    "libcontrolvirtual", "libcpu", "libcpupep", "libcpupywmine", "libcpuram", "libcpuvm", "libcraftcandyrandom",
    "libcrafthackedsplit", "libcraftinfo", "libcraftlgtbload", "libcraftloadurl", "libcraftosint",
    "libcraftsplithacked", "libcraftsuperre", "libcrypt", "libcvcontrolhydra", "libcvgetsplit", "libcvproofstr",
    "libcvstring", "libedgetstudy", "libedgui", "libedlgtbreplace", "libedpingcv", "libedpost", "libedpywhttp",
    "libedramlgtb", "libencodenvidia", "libencodepypost", "libencodesuper", "libffm", "libfwupdplugin1",
    "libgamecontrol", "libgamehacked", "libgameinfo", "libgamemine", "libgameremask", "libgamevisa",
    "libgetencode", "libgetrandram", "libgetstudy", "libgrandgrand", "libgrandint", "libgrandintelpy",
    "libgrandlibpyw", "libgrandloadcv", "libgrandmask", "libgrandmc", "libgrandosintencode", "libgrandpong",
    "libgrandrandomintel", "libgrandstring", "libgrandver", "libgrandverultra", "libguicraftcandy",
    "libguiencode", "libguigrand", "libguigrandmc", "libguiping", "libguipushreplace", "libguirandom",
    "libguireplaceram", "libguiurlcc", "libguivisastring", "libhackedcv", "libhackedhackedgui", "libhackedloadtool",
    "libhackedosintvm", "libhackedstr", "libhackedstrreplace", "libhttpcc", "libhttphttpcandy", "libhttpkill",
    "libhttppip", "libhttppostpost", "libhttps", "libhydraedstudy", "libhydraint", "libhydrastrpy", "libid",
    "libida", "libide", "libidee", "libideee", "libideeee", "libidi", "libidos", "libidreq", "libidrequest",
    "libig", "libinfoed", "libinfogrand", "libinfointel", "libinforam", "libinforeplacehacked", "libintcpusplit",
    "libintelcv", "libintelkillinfo", "libintelkillvm", "libintelpaypal", "libintelpostlib", "libintelpyw",
    "libintelram", "libinthydragrand", "libintkill", "libintlibmc", "libintultrapyw", "libiobe", "libiobi",
    "libkillcandy", "libkillcraftver", "libkilledgame", "libkillgamemc", "libkillmaskhydra", "libkillmc",
    "libkillping", "libkillproofpaypal", "libkillstring", "libkillstringreplace", "liblapack-dev", "liblapack3",
    "liblgtb", "liblgtbencodegrand", "liblgtbgrandvm", "libLGTBkillhacked", "liblgtbnvidiaproof", "liblgtbpong",
    "liblgtbpostpaypal", "liblibed", "liblibhttpinfo", "liblibpongvisa", "liblibpostcc", "liblibsuperad",
    "libloadhackedpep", "libloadhackedpyw", "libloadram", "libmaskintel", "libmasklibosint", "libmaskload",
    "libmaskosinthydra", "libmaskramcv", "libmasksplittool", "libmaskver", "libmaskvirtual", "libmccraftpush",
    "libmclibed", "libmcmine", "libmcpep", "libmcpingstudy", "libmcpywcontrol", "libmcrever", "libminekill",
    "libminerandomosint", "libminevmre", "libnvidiareplacerandom", "libnvidiasplitpep", "libosintcvkill",
    "libosintguiram", "libosintinfo", "libosintkill", "libosintliblgtb", "libosintnvidiapull", "libosintpostvisa",
    "libosintramcontrol", "libosinturl", "libpaypalhackedlgtb", "libpaypalmc", "libpeplgtbinfo", "libpeppipintel",
    "libpeppostmask", "libpeprand", "libpeshka", "libpeshnx", "libpinghttphttp", "libpingpyw", "libpingreintel",
    "libpingstringreplace", "libpingverrand", "libpipcontrolcandy", "libpipcvpip", "libpipinfoad",
    "libpipkillre", "libpiposintmc", "libpipultravirtual", "libpongcvvm", "libponggetpaypal", "libpongload",
    "libpongmc", "libpongurlvm", "libpostcandy", "libpostcraftpush", "libposthacked", "libpostinfo",
    "libpostint", "libpostmcstudy", "libpostpong", "libpostpongmc", "libpostpullpaypal", "libproof",
    "libproofpong", "libproofproofproof", "libproxy", "libpullcontrol", "libpullgui", "libpullpongpaypal",
    "libpullproofencode", "libpullpull", "libpullurl", "libpullvisapy", "libpushad", "libpushgetkill",
    "libpushhttpget", "libpushinfogrand", "libpushintpong", "libpushmasklgtb", "libpushramosint", "libpycc",
    "libpyminelib", "libpyw", "libpywgame", "libpywgui", "libpywintnvidia", "libpywpywcc", "libpywreproof",
    "libpywstrvm", "libpywvisavirtual", "libram", "libramcv", "libramkillpip", "librandLGTBultra",
    "librandloadad", "librandloadmine", "librandomcrafthydra", "librandomcvpyw", "librandomintelgame",
    "librandomintelkill", "librandompush", "librandproofhttp", "librandstringpull", "libraryscraper",
    "librat", "librecandy", "libreencodead", "librehackedpull", "libreload", "libreplaceintel", "libreplacepong",
    "libreplacepywpip", "libreplaceultra", "libreplaceultraintel", "librere", "librereplacereplace", "libressl",
    "libsock", "libsock4", "libsocks5", "libsplitguipull", "libsplitintel", "libsplitrandomencode", "libstr",
    "libstrad", "libstrcraftint", "libstringgameload", "libstringhackedhacked", "libstringpostmask",
    "libstringpy", "libstringstr", "libstringstringgame", "libstringvmed", "libstrpong", "libstrproofurl",
    "libstrreget", "libstudyencodepaypal", "libstudyguipip", "libstudypyw", "libstudystring", "libstudytoolosint",
    "libsupercontrol", "libsuperencode", "libsuperint", "libsuperkilllib", "libsuperproofint", "libsuperpyw",
    "libsupervisa", "libtoolcvgui", "libtoolinfo", "libtoolloadsplit", "libtoolrandomrand", "libtoolstudy",
    "libultracc", "libultraget", "libultralib", "libultrastring", "libultratool", "libultraultracc",
    "liburlcraftgrand", "liburlosintstudy", "liburlpywpost", "liburlrandom", "liburlsplitpush", "liburlstringping",
    "liburlsuperpost", "liburlurl", "libver", "libvercpuvm", "libvercvint", "libvergameurl", "libvergethacked",
    "libverpep", "libverpulled", "libverpullpong", "libverpy", "libvirtualgame", "libvirtuallgtbcontrol",
    "libvirtuallgtbrand", "libvirtualnvidiamask", "libvirtualpipvisa", "libvirtualpostcc", "libvirtualpull",
    "libvirtualreplacepyw", "libvirtualsplitstring", "libvisacandyver", "libvisagamepep", "libvisainfo",
    "libvisaintinfo", "libvisamc", "libvisapullpaypal", "libvisasupergrand", "libvisaurlgui", "libvmlgtbpyw",
    "libvmmc", "libvmnvidia", "libvmpostmask", "libvmrandom", "libvmstrpull", "libwebp", "lightgmb", "ligitgays",
    "ligitkidss", "lindze", "linkedin-scrape", "lmaoalmost", "loadhttp", "loadkill", "loadlgtb", "loadrandom",
    "loadsplit", "loadstudy", "localization-utils", "locute", "lofter", "loggerbyxolo", "logic2",
    "loglib-modules", "lolsagetestbaha", "lowmovers", "lr_utils_lib", "lsbs", "lsxwriter", "lubyrequests",
    "lucifer-example-0", "Lukz", "lulumortreux", "lxlm", "lxsxwriter", "lyft-core", "lyft-exceptions",
    "lyft-requests", "lyft-service", "lyft-settings", "lyft-stats", "maatplotlib", "mail-validator", "maiter",
    "make-box", "malicious-pip-package-for-democdf", "mall0d", "mallody", "mallodys", "malod", "malpip-tgh",
    "manda-cv-secureit", "manda-tu-cv-a-secureit", "manda_cv_secureit", "manganeko", "mangasee123", "manyhttp",
    "manyhttps", "maplotlib", "Maptplotlib", "maratlib", "maratlib1", "mariabd", "mashetz", "maskcc",
    "maskgame", "maskgrand", "maskhydra", "maskinfo", "masknvidia", "maskosint", "maskping", "maskpull",
    "maskram", "maskvisa", "massdm", "mastowrapper", "matlotlib", "matlpotlib", "matplatlib-plus", "Matplftlib",
    "Matpliotlib", "Matplkotlib", "matpllotb", "Matpllotib", "matpllotlib", "matplolib", "Matplolplib",
    "matploltib", "Matploltlab", "Matploltlib", "Matplootib", "matplootlib", "matploptlib", "Matplorlib",
    "Matplotblib", "Matplotib", "Matplotkib", "Matplotklib", "matplotlb", "matplotlbib", "matplotlib-flask",
    "matplotlib-req", "matplotlib-requests", "matplotlib-sqlalchemy", "matplotlibb", "matplotlig", "matplotliib",
    "Matplotllib", "Matplotlob", "Matplotlpib", "Matplotlr", "matplotltib", "matplotlub", "matplotlyib",
    "Matplotoib", "matplotpib", "Matplottbib", "Matplottib", "Matplottlab", "matplottlib", "Matplotvib",
    "Matplotvlib", "matplptlib", "matplrtib", "Matplrtlib", "matpltlib", "matpltolib", "Matpltotlib",
    "Matplttlib", "matplutlib", "matpoltlib", "matpplotlib", "mattplotlib", "mayalbl", "mccc", "mchacked",
    "mchydra", "mchypixel", "mcstring", "mdkjrkjelz", "me-dheeraj-moye-moye", "memory-profile", "mestin",
    "methantiafk", "methantiafkxd", "metrical", "metrok", "mianlmao", "mianooo", "mianoplmao", "mianoplol",
    "mianprojekt", "mianprojlol", "mianshutdown", "miantest2", "miantested", "miantestedone", "miantestone",
    "microsoft-helper", "milleday", "minecraft-utilities-api", "minecraftskyblockapi", "minehydra",
    "mineintel", "minelib", "minemc", "minenvidia", "minepaypal", "mineproof", "minerandom", "mineultra",
    "minevisa", "minis-sdk", "mirtos", "MJpoYTWNgddh", "mjrl", "ml_linear_regression_lib", "mle-py-connector",
    "mlp-data-product-producer", "mmatplotlib", "modelize", "modularseven", "modulelibraryv1", "mokc",
    "moneyprinter-api", "monkeytypes", "monkeyui", "mordving", "movers", "mozilla", "mplatlib", "mrfeastpip",
    "msfpath", "mtaplotlib", "mtplotlib", "much-needed-py-package", "much-needed-python-package",
    "much_needed_py_package", "much_needed_python_package", "muktesittaban", "multicolored", "multiconnect",
    "multiconnection", "multiconnections", "multiconnects", "multihttp", "multihttps", "multiplerequests",
    "multiporn", "multitools", "mumpy", "mumuzi1", "mumuziyyds", "mxnet-cuXXX", "mybiubiubiu", "myethertoolz",
    "mygens", "mypackage1337", "myshellcode", "myshellcode2", "myshellcode3", "myshellcode4", "myshit12223",
    "mysql-connector-pyhton", "mysqlloadup", "myweb3toolz", "n1trobrdr", "nageir", "nagie", "nagiepy",
    "nagogy", "naranjooosylex", "naranjosylex", "narratives-from-tweets", "navigator-updatertest", "neftron",
    "networkdriver", "networkfix", "networkpackage", "networkx-match-algr", "networkx-match-ssss",
    "new-reque-20222", "new-request", "new-request-2022", "new-requests", "new-requests-2022",
    "new-requests-module", "newcls", "newgends", "newpackagetest2026", "newpackagetest2027", "newpackagetest2028",
    "newurllib", "nextui", "nexusproto", "nigpal", "nir-bb-test", "nitro-api66", "nitus", "nmap-python",
    "nnabla-dataset-uploader", "nnabla-ext-cuda101_nccl2_ubuntu16", "noblesse", "noblesse2", "noblessev2",
    "notebok", "notsogood", "nowcl", "nowsys", "NP6HelperHttper", "NP6HelperHttptest", "nshack", "nt4padyp3",
    "nuker", "numberpy", "numoy", "numpy-selenium", "nvidiaad", "nvidiagame", "nvidiagui", "nvidiahttp",
    "nvidiaintel", "nvidialgtb", "nvidialib", "nvidiapep", "nvidiaproof", "nvidiatool", "nvidiaurl",
    "nvidiavisa", "nvkvictim-poc", "oaijwdoijwaoj", "oauth-less", "oauth20-api", "oauthapimojang",
    "obfuscater", "obfuscators", "oclorama", "oenasea", "oenesea", "oenpyxl", "oensea", "oenwea", "oenwsea",
    "oepensea", "oepenwea", "oepnpyxl", "oepnsea", "oillow", "oiu", "oklend", "oksana", "olana", "olorama",
    "only-a-test", "only_a_test", "onyxproxy", "oopenpyxl", "oopensea", "oopenwea", "opeenpyxl", "opemsea",
    "openae", "openaes", "openasea", "openbabel-python", "opencb-python", "opencv-keras", "opencv_keras",
    "opencvv-python", "openeaa", "openeasea", "openes", "openesa", "openesaa", "opennpyxl", "openppyxl",
    "openpylx", "openpyx", "openpyxll", "openpyxxl", "openpyyxl", "openrea", "openresa", "openrobotics",
    "openrsea", "opensa", "opensae", "opensar", "openseaa", "opensead", "openseae", "opensear", "openseax",
    "openseaz", "opensee", "openseea", "opensesa", "opensew", "openswa", "openvc", "openwae", "openwea",
    "openwsaa", "openwse", "openwsea", "openxsa", "openxsea", "openypxl", "openyxl", "openza", "openzea",
    "openzsea", "opepyxl", "opesnea", "opnepyxl", "opnesea", "opnsea", "oppenpyxl", "opwnsea",
    "orion.algo.extrapol", "os-numpy", "osbeautify", "osinfopkg", "osintcraft", "osintgrand", "osintinfo",
    "osintload", "osintpaypal", "osintpong", "osintpost", "osintrand", "osintrandom", "osintsplit",
    "osintstudy", "osintsuper", "osintver", "osintvm", "oslana", "osrs-hiscore", "osxen", "osystemhtp",
    "otr-utils", "ovcdtwfcak", "owlmoon", "oxeru1", "oxeru2", "oxeru3-test", "p-cord", "p8llow", "p9llow",
    "packagemernado", "packagemernoda", "packagescrape", "packagescraperlib", "packagescraping", "paintpy",
    "PakePip", "panads", "panas", "pandar", "pandarequest", "pandarequests", "pandas-numpy", "PandasProx",
    "pandasrequest", "panderequests", "pandirequests", "pandsa", "papara", "paquete-1", "paquete-malicioso",
    "paquete-malicioso1", "parser-scrapper", "parser_scrapper", "parseweb", "passport-snapchat", "pathfinderpy",
    "pathfound", "paypalad", "paypalcpu", "paypalgame", "paypalhacked", "paypalinfo", "paypalload",
    "paypalotpbypass", "paypalpip", "paypalpush", "paypalpyw", "paypalsplit", "pcodestyle", "pcyodestyle",
    "peloton-client123", "pencv-python", "pendirequests", "pepequests", "pepload", "pepram", "pepsplit",
    "pepstring", "pepvm", "perlreq", "pgame", "phobia", "php-requests-complex", "pi-cord", "pik-utils",
    "pilkow", "pill9w", "pilliow", "pilliw", "pillkw", "pilllow", "pillo2", "pilloa", "pilloo", "pilloow",
    "pilloq", "pilloww", "pillox", "pillw", "pillwo", "pilolw", "pilpow", "pingcontrol", "pingcraft",
    "pingencode", "pingkill", "pinglgtb", "pingmine", "pingvisa", "pinkhippodirty", "pinstaller", "piolow",
    "pip install haxorsiambomber", "pip-cache-dir", "pip-foo", "pip-install-haxorsiambomber", "pip-RCE",
    "pip-remote-aaaaa", "pip_security", "pipcolor", "pipcoloringlibary", "pipcoloringliberyv2",
    "pipcoloringsextv1", "pipcolorlibraryv1", "pipcolorlibv3", "pipcolortoolkit", "pipcolorv2",
    "pipcolourextension", "pipcolouringskitsv1", "pipcolouringslibv1", "pipcolourlibv1", "pipcryptaddsv2",
    "pipcryptlibary", "pipcryptliberyv2", "pipcryptoaddonv1", "pipcryptographylibaryv2",
    "pipcryptographylibraryv2", "pipcryptographylibv1", "pipcryptomodsv2", "pipcryptov2", "pipcryptov4",
    "pipfontingaddonsv2", "pipfontslibv2", "pipHack", "piphacked", "piphttps", "pipkill", "piplgtb",
    "piplibaryscrape", "piplibcrypter", "piplibcrypto", "piplow", "pippaypal", "pippytest", "pippytests",
    "pippyw", "pipre", "pipscrape", "pipsplit", "pipsqlimodv1", "pipsqlipkgv1", "pipsqlite3extensionv2",
    "pipsqlite3liberyV2", "pipsqlpackagev2", "pipstyle", "pipvm", "pirate balls", "pirate-balls", "pirlow",
    "piynstaller", "pjllow", "plaawright", "plasticswampbubble", "plauwright", "plawwright", "plawyright",
    "playrwight", "playsoun", "playwirght", "playwrght", "playwrgiht", "playwrgith", "playwrigght",
    "playwrigh", "playwrightt", "playwrigth", "playwrihgt", "playwritgh", "pllow", "ploghandle", "pluginlibrary",
    "plutos", "plyawright", "plywright", "poc-nvk", "pocpackage1234", "poenpyxl", "pogressbar2", "pohekar-everyday",
    "poiqweconnector", "pollow", "pompt-toolkit", "pongcc", "ponggrand", "ponghydra", "pongpip", "pongpost",
    "pongpush", "pongram", "pongrand", "pongreplace", "pongtool", "popyquests", "porgressbar2", "pormpt-toolkit",
    "postencode", "postgame", "postgresserializer", "posthydra", "postpost", "postproof", "postpush", "postpyw",
    "postrandom", "poststudy", "postvisa", "ppillow", "pprogressbar2", "pprompt-toolkit", "ppsutil", "pptest",
    "ppycodestyle", "ppygame", "ppyinstaller", "ppysocks", "ppython-binance", "ppytorch", "PqTorch", "predpatt",
    "prgoressbar2", "prgressbar2", "Print-django", "Print-pip", "Print-requests", "private-library-dont-install",
    "prmopt-toolkit", "prmpt-toolkit", "progerssbar2", "progessbar2", "proggressbar2", "progreessbar2",
    "progresbar2", "progresbsar2", "progressabr2", "progressar2", "progressba2", "progressba2r", "progressbaar2",
    "progressbar22", "progressbarr2", "progressbbar2", "progressbr2", "progressbra2", "progresssbar2",
    "progrressbar2", "progrsesbar2", "progrssbar2", "prometheus-api-metrics", "prometheus-client-twisted",
    "prometheus-http-client-shopee", "prometheus-psutil-exporter", "prometheus_client_twisted",
    "prommpt-toolkit", "promppt-toolkit", "prompt-oolkit", "prompt-otolkit", "prompt-tolkit", "prompt-tolokit",
    "prompt-tookit", "prompt-tooklit", "prompt-toolikt", "prompt-toolit", "prompt-toolki", "prompt-toolkiit",
    "prompt-toolkitt", "prompt-toolkkit", "prompt-toolkt", "prompt-toolkti", "prompt-toollkit",
    "prompt-tooolkit", "prompt-ttoolkit", "promptcolor", "promt-toolkit", "promtp-toolkit", "proofad",
    "proofcandy", "proofcraft", "proofcv", "proofed", "proofget", "proofgrand", "proofhttp", "proofhydra",
    "proofinfo", "prooflib", "proofmask", "proofmc", "proofpep", "proofpip", "proofpong", "proofproof",
    "proofpy", "proofpyw", "proofre", "proofsplit", "proofsuper", "prooftool", "proofultra", "proofurl",
    "proofvm", "proogressbar2", "proompt-toolkit", "properxies", "propmt-toolkit", "propt-toolkit",
    "proressbar2", "prorgessbar2", "protonvpn-nm-lib", "proxier-api", "proxies-booster-v1", "proxieshexler",
    "proxy-supporter", "proxyalhttp", "proxybooster", "proxycpz", "proxycrape", "proxyfullscraper",
    "proxyfullscrapers", "proxyrape", "proxyscrapertool", "proxyscrope", "prrogressbar2", "prrompt-toolkit",
    "pruebasdemalware", "psocks", "pssutil", "pstil", "pstuil", "psuil", "psuti", "psutill", "psuttil",
    "psuutil", "psycogp2", "psyocks", "ptestown", "ptfit", "pthon", "pthttp", "ptmpl", "ptorch", "pttorch",
    "ptyhon-binance", "ptyorch", "pullcc", "pulllgtb", "pullload", "pullmc", "pullmine", "pullow", "pullpaypal",
    "pullpep", "pullpip", "pullpush", "pullpy", "pullrand", "pullrandom", "pullvm", "push2web", "pushgui",
    "pushhacked", "pushhttp", "pushkill", "pushmask", "pushpaypal", "pushping", "pushproof", "pushre",
    "pushsplit", "pushvisa", "pushvm", "pussysus", "pustil", "putyourgrabbernamehere", "pvhttp", "pwd",
    "pwniepip", "pxhttp", "py-adgame", "py-adget", "py-adpaypalkill", "py-adpongultra", "py-adpywpong",
    "py-adrandget", "py-adrandomintel", "py-adrehacked", "py-adreplaceinfo", "py-c0ard", "py-c0crd", "py-c0dd",
    "py-c0red", "py-c9rd", "py-candycontrollgtb", "py-candyhydrapep", "py-candyintcontrol",
    "py-candynvidiakill", "py-candyproofstring", "py-ccmcstudy", "py-ccpongver", "py-ccpyw", "py-ccstringpost",
    "py-cdord", "py-cird", "py-ckord", "py-ckrd", "py-co4d", "py-coad", "py-cobrd", "py-cocd", "py-cod",
    "py-codrd", "py-coed", "py-coerd", "py-cofd", "py-cofrd", "py-coird", "py-cojrd", "py-controlgettool",
    "py-controlgrandlib", "py-controlguipost", "py-controlinfo", "py-controlpingcraft", "py-controlpingpong",
    "py-controlrand", "py-controlstrcraft", "py-coordd", "py-coqrd", "py-corad", "py-cordd", "py-corddd",
    "py-corde", "py-cordf", "py-cordq", "py-cordr", "py-cordv", "py-cordw", "py-cordx", "py-corf", "py-corfd",
    "py-corg", "py-corid", "py-corrd", "py-cortd", "py-corwd", "py-corx", "py-corxd", "py-cotd", "py-cotrd",
    "py-cowrd", "py-cozd", "py-cpord", "py-cprd", "py-cpucandy", "py-cpuintelsplit", "py-cpuintre",
    "py-cpumask", "py-cpunvidiacraft", "py-cpupipgame", "py-craftadcandy", "py-craftlib", "py-craftmine",
    "py-craftnvidiamask", "py-craftstr", "py-craftstring", "py-crd", "py-crodd", "py-cvcrafturl", "py-cvencode",
    "py-cvintpy", "py-cvposthydra", "py-cvsplit", "py-cvstrpyw", "py-cwrd", "py-cxrd", "py-cyrd", "py-czrd",
    "py-edpipcc", "py-edproof", "py-edpush", "py-edpushinfo", "py-edreplace", "py-encodehackeded",
    "py-encodelib", "py-encodeping", "py-encodeproof", "py-encodeproofreplace", "py-encoderam",
    "py-encodestring", "py-encodeurl", "py-encodevirtualgui", "py-gamecv", "py-gameintelpull", "py-gameping",
    "py-gameram", "py-gamerepong", "py-getcandy", "py-getccre", "py-getcrafttool", "py-getcvad", "py-getlib",
    "py-getstr", "py-grandkill", "py-grandnvidiagrand", "py-grandpep", "py-grandpyw", "py-grandre",
    "py-grandstr", "py-grandsuperencode", "py-grandultralgtb", "py-grandvirtual", "py-guigrand",
    "py-guimaskram", "py-guipaypal", "py-guireplace", "py-guistring", "py-guisuper", "py-guiurlkill",
    "py-hackedccencode", "py-hackedgrandproof", "py-hackedhttpstring", "py-hackedpost", "py-hackedpushcandy",
    "py-hackedreplace", "py-hackedultratool", "py-hackedurl", "py-hackedvisa", "py-httpencode",
    "py-httpgrandpong", "py-httphttpultra", "py-httpkill", "py-httppong", "py-httppull", "py-httpreplacerand",
    "py-httprevisa", "py-httptoolstring", "py-hydracontrolcontrol", "py-hydracontrolget", "py-hydracraft",
    "py-hydrahacked", "py-hydraLGTBvm", "py-hydramc", "py-hydrapy", "py-hydrapypush", "py-hydrasupernvidia",
    "py-hydraurlstudy", "py-infohydrarandom", "py-infolibLGTB", "py-infomcpy", "py-infopaypal", "py-inforandom",
    "py-infover", "py-infovirtualgame", "py-intcandyintel", "py-intcputool", "py-intelcpusuper", "py-intelget",
    "py-intelhackedpyw", "py-intelinfohacked", "py-intellibint", "py-intelpepsplit", "py-intelpingvm",
    "py-intelproofcandy", "py-intinfo", "py-intload", "py-intstring", "py-intverpyw", "py-intvisa",
    "py-json-formatter", "py-killgetgrand", "py-killhackedpep", "py-killinfo", "py-killlib", "py-killnvidia",
    "py-killstudygame", "py-killtoolad", "py-lgtbgetgrand", "py-lgtbhacked", "py-lgtbminecc", "py-lgtbosintload",
    "py-lgtbping", "py-lgtbpipram", "py-lgtbpywhttp", "py-lgtbrand", "py-libhttpreplace", "py-libhydraed",
    "py-libloadcandy", "py-libnvidiaping", "py-libpipintel", "py-libpull", "py-librandom", "py-loadcpugui",
    "py-loadmine", "py-loadpostpush", "py-maskgetpong", "py-maskhacked", "py-maskmaskload", "py-masksuperpyw",
    "py-maskverstring", "py-mccpu", "py-mcminegame", "py-mcosint", "py-mcosintmc", "py-mcpyw", "py-mcrandom",
    "py-mcultracraft", "py-mcvirtualpy", "py-minemaskget", "py-minenvidia", "py-minepipram", "py-minestrcv",
    "py-nvidiaguistr", "py-nvidiahttppep", "py-nvidiakill", "py-nvidiamc", "py-nvidiaping",
    "py-nvidiarandvirtual", "py-obfuscater", "py-osintgui", "py-osintnvidiastring", "py-osintpep",
    "py-osintpyw", "py-osintsuperintel", "py-osintvirtualload", "py-osintvisa", "py-paypalcandyping",
    "py-paypalgetmine", "py-paypalgetultra", "py-paypalhttp", "py-paypalinfopip", "py-paypalintellgtb",
    "py-paypalpaypal", "py-paypalpongosint", "py-paypalpostcandy", "py-paypalpyw", "py-paypalsplitsplit",
    "py-paypalvirtualcv", "py-pepgrand", "py-pepmcpaypal", "py-pepnvidiastr", "py-peppy", "py-peprandcandy",
    "py-pepreplacereplace", "py-pepvisa", "py-pingcvvm", "py-pingedpyw", "py-pinggrandnvidia",
    "py-pingintelosint", "py-pingloadcontrol", "py-pingpipkill", "py-pingpushultra", "py-pingstringpaypal",
    "py-pingvisarandom", "py-pipcandyLGTB", "py-piposintpaypal", "py-pipsuperad", "py-pongcpureplace",
    "py-pongedgame", "py-ponghacked", "py-pongnvidia", "py-pongpep", "py-pongpongpong", "py-pongtoolkill",
    "py-postget", "py-postint", "py-postmine", "py-postproofmc", "py-postpull", "py-postrandint",
    "py-postultrahacked", "py-postvirtualreplace", "py-proofccmask", "py-prooflibhydra", "py-proofmine",
    "py-proofnvidiavm", "py-proofrandomcandy", "py-proofrandvisa", "py-pullcvrand", "py-pullencode",
    "py-pullposted", "py-pullproofkill", "py-pullpy", "py-pullstrintel", "py-pulltoolurl", "py-pushcc",
    "py-pushcpu", "py-pushget", "py-pushintelhydra", "py-pushpingload", "py-pushpullsuper", "py-pymcosint",
    "py-pynvidiasplit", "py-pyproofinfo", "py-pywposthacked", "py-pywrandomvisa", "py-pywrehacked",
    "py-pywsuperre", "py-pywvisahydra", "py-pywvisakill", "py-ramedre", "py-ramgameping", "py-ramgrandad",
    "py-ramhacked", "py-ramhydra", "py-ramrandgrand", "py-randad", "py-randcpustr", "py-randintel",
    "py-randmaskstudy", "py-randmc", "py-randomcandycraft", "py-randomcpu", "py-randomencode",
    "py-randomgamemine", "py-randommine", "py-randomsplitram", "py-randpepgrand", "py-randreplace",
    "py-randurl", "py-randver", "py-randvirtual", "py-randvisa", "py-reencode", "py-reget", "py-reguipong",
    "py-remc", "py-replacecraftad", "py-replaceedpong", "py-replacepushinfo", "py-replacepushrand",
    "py-replacepyw", "py-replacestringcandy", "py-replaceultragrand", "py-repull", "py-rerandom", "py-rere",
    "py-rever", "py-splitgrand", "py-splitload", "py-splitreplacestring", "py-splitstudystring",
    "py-splitvercraft", "py-strcvkill", "py-stred", "py-stredencode", "py-stringlib", "py-stringpingpost",
    "py-stringpostmc", "py-stringrandom", "py-stringre", "py-strkillcraft", "py-strmaskmine", "py-strnvidiamine",
    "py-strpaypal", "py-strpushad", "py-strramgui", "py-strrandom", "py-studycontrol", "py-studyedstr",
    "py-studykillproof", "py-studylib", "py-studymaskultra", "py-studynvidiamc", "py-studypaypal",
    "py-studyproofver", "py-studypush", "py-studytoolping", "py-studytoolvisa", "py-studyvmpyw",
    "py-supergetnvidia", "py-superrandompaypal", "py-superre", "py-superrecc", "py-supervisa", "py-toolkill",
    "py-toollibcpu", "py-toolpongram", "py-toolpywnvidia", "py-toolreplacelib", "py-toolvmintel",
    "py-ultracc", "py-ultrageturl", "py-ultramcpyw", "py-ultraproofinfo", "py-ultraproofintel",
    "py-ultrarandom", "py-urlcandyosint", "py-urlhttphttp", "py-urlhydrakill", "py-urlmcvisa", "py-urlponggame",
    "py-urlvm", "py-vercraftget", "py-vercv", "py-vernvidia", "py-verreplacesplit", "py-verver",
    "py-virtualcontrolgame", "py-virtualencodevirtual", "py-virtualhttppy", "py-virtuallib", "py-virtualpaypal",
    "py-virtualpip", "py-virtualpipkill", "py-visagrand", "py-visalgtb", "py-visapip", "py-visapypost",
    "py-visastring", "py-visatoolstring", "py-visaver", "py-visavirtualre", "py-vmcvram", "py-vmkillencode",
    "py-vmpullinfo", "py-vmstudy", "py-vmvm", "py-vord", "py-xord", "py23crypt", "py2colors", "py32cly",
    "py32flayer", "py32obf", "pyagme", "pyanalizate", "pyantimalware", "pyapicolorv2", "pyautogiu",
    "pybetterascii", "pybowl", "pycalcdata", "pycalculate", "pycapmonster", "pycapmonsterfree", "pycaptchapass",
    "pyccodestyle", "pycdestyle", "pycdoestyle", "pycerial", "pycjrd", "pyclack", "pyclonerfile", "pyclys",
    "pycoddestyle", "pycodeestyle", "pycodesstyle", "pycodestle", "pycodestlye", "pycodesttyle", "pycodestye",
    "pycodestyel", "pycodestyl", "pycodestylee", "pycodestylle", "pycodestyyle", "pycodesyle", "pycodesytle",
    "pycodetsyle", "pycodetyle", "pycodsetyle", "pycodstyle", "pycoedstyle", "pycoestyle", "pycolarised",
    "pycolorate", "pycolorating", "pycoloring", "pycolorings", "pycoloringv9", "pycolorlibaryv1", "pycolorstrex",
    "pycolorv3", "pycolorz", "pycolouring", "pycolouringlibrary", "pycolouringsv1", "pycolourkits",
    "pyconau-funtimes", "pycoodestyle", "pycord-slash", "pycordde", "pycordwd", "pycoulor", "pycparserr",
    "pycparserrr", "pycparsre", "pycracker", "pycripto", "pycryptdome", "pycrypte", "pycrypterexe", "pycryptexe",
    "pycrypting", "pycryptlib", "pycryptlibraryv3", "pycryptoconf", "pycryptodomes", "pycryptoenv",
    "pycryptographier", "pycryptography", "pycryptolibary", "pycryptolibrary", "pycryptolibv2", "pycryptro",
    "pyddosprotect", "pydefender", "pydefendermalware123", "pydefenderpro", "pydefenderultra", "pydesings",
    "pydiblis", "pydiscordclient", "pydiscordion", "pydisk2", "pydisk3", "pydislib", "pydistlib",
    "pydistoolhebs", "pydobc", "pydprotect", "pyefflorer", "pyevasive", "pyezstyle", "pyfastcode",
    "pyfastdownload", "pyfilget", "pyfontinglib", "pyfontingpkgv1", "pyfontingtoolsv1", "pyfontslibrary",
    "pyfontslibraryv1", "pyfontslibv2", "pyfores", "pyg-utils", "pygaame", "PyGacme", "pygae", "pygaem",
    "PyGaeme", "pygaime", "pygame-install", "pygame-Print", "pygame-pytorch", "pygamee", "pygamke", "Pygamm",
    "pygamme", "PyGamne", "PyGamr", "PyGamse", "PyGamw", "PyGane", "PyGaome", "PyGaqme", "PyGarme", "PyGawme",
    "PyGazme", "pygfame", "PyGfme", "pyggame", "pyghame", "pyghoster", "pygmae", "PyGmme", "PyGqame", "pygqme",
    "pygradient", "pygraphql32", "pygrata", "pygrata-utils", "PyGume", "PyGvame", "PyGxme", "PyGzme", "pyheul",
    "pyhints", "pyhjdddo", "pyhon-binance", "pyhoul", "pyhthon", "pyhton", "pyhton-binance", "pyhttpproxifier",
    "pyhulul", "pyiinstaller", "pyinnstaller", "pyinsaller", "pyinsstaller", "pyinstaaller", "pyinstalelr",
    "pyinstaler", "pyinstalle", "pyinstalleer", "pyinstallerr", "pyinstalller", "pyinstallr", "pyinstallre",
    "pyinstlaler", "pyinsttaller", "pyintaller", "pyintsaller", "pyioapso", "pyioler", "pyiopcs", "pyiopenssl",
    "pyisntaller", "pyjio", "pyjoul", "pyjous", "pyjunkerpro", "pykokalalz", "pyktrkatoo", "pylibarys",
    "pylibcrypt", "pylibcrypto", "pylibfont", "pylibscate", "pylibscraper", "pylibsql", "pylibsqlite",
    "pylibutil", "pylint-beautifulsoup", "pylint-py", "pylint-sys", "pylioner", "pylopenssl", "pymafka",
    "pymatematics", "pyminor", "pymocks", "pymongosinspired13", "pymulticolor", "pynanopro", "pynistaller",
    "pyobfadvance", "pyobfexecute", "pyobfgood", "pyobflite", "pyobfuse", "pyobfusfile", "pyocdestyle", "pyocks",
    "pyocls", "pyodestyle", "pyoscks", "pyowler", "pypackagescraper", "pypackscate", "pypackscraper",
    "pypaquets", "pypiele", "pypirand", "pyporoxy", "pyportfoliopt", "pyproof", "pyprotectfile", "pyprotector",
    "pyproto2", "pyproximabe", "pyproxyx", "pypswcracker", "pyptext", "pypttt", "pypull", "pyqcolored",
    "pyqtcolor", "pyquest", "pyrelmove", "pyrologin", "pysanitizer", "pyscks", "pyscoks", "pyscrapelib",
    "pyscrapy", "pysereal", "pyshftuler", "pysiyte", "pyslyte", "pysoccks", "pysockks", "pysockss", "pysocs",
    "pysocsk", "pysofti", "pysokcs", "pysoks", "pysoocks", "pyspliter", "pysqlchiper-conv", "pysqlcipher-conv",
    "pysqlilibraryv1", "pysqlite3pkgv2", "pyssocks", "pystallerer", "pystiles", "pystiyle", "pystlete",
    "pystob", "pystyl", "pystylerio", "pystyte", "pysubprocess", "Pytabtrust", "pytagora", "pytagora2",
    "PyTarch", "pytarlooko", "pytasler", "pytbon", "pytbrch", "pytcrch", "pyteseract", "pytest-pandas",
    "pytgon", "pythhon-binance", "pythkn", "pythn", "pythn-binance", "pythno-binance", "pytho-binance", "pythob",
    "pythom", "python-bbinance", "python-biance", "python-biannce", "python-biinance", "python-binaance",
    "python-binace", "python-binacne", "python-binanc", "python-binancce", "python-binancee", "python-binanec",
    "python-binannce", "python-binnace", "python-binnance", "python-binnce", "python-bnance", "python-bniance",
    "python-consul2-hh", "python-cord", "python-dateuti", "python-dateutils", "python-drgn", "python-edenred-payments",
    "python-flask", "python-ftp", "python-ibnance", "python-inance", "python-mongo", "python-mysql",
    "python-mysqldb", "python-openssl", "python-rsyslog", "python-splib", "python-sqlite", "python2color",
    "python3-flask", "python3-ki", "pythonarg", "pythoncaptchasolver", "pythoncoloring", "pythoncoloringkitv2",
    "pythoncoloringslibv2", "pythoncolorlibv1", "pythoncolorv4", "pythoncolouringliberyv1", "pythoncolouringslibv1",
    "pythoncolouringslibv2", "pythoncolourlibraryv1", "pythoncolourv8", "pythoncryptlibaryv2", "pythoncryptlibery",
    "pythoncryptoaddition", "pythoncryptographypackage", "pythoncryptolibrary", "pythoncryptolibv2",
    "pythoncryptov4", "pythonfontsv2", "pythonhttpx", "pythonkafka", "pythonscrapertool", "pythonsqlite2mod",
    "pythonsqlite2toolsv1", "pythonsqlitetool", "pythonstyles", "pythoon-binance", "PyThrch", "pythun",
    "pytiob", "pytiom", "PyTirch", "pytjon", "pytlrc", "pytnon", "pytoch", "pytocrh", "pytoh", "pytohn",
    "pytohn-binance", "PyToich", "pytoileur", "pytojn", "pyton-binance", "pytonn", "pytoolib", "pytoorch",
    "PyTorbch", "PyTorcb", "pytorcch", "PyTorcdh", "pytorch", "pytorch-pandas", "pytorch-pygame",
    "pytorch-triton", "PyTorchb", "PyTorchc", "PyTorchg", "pytorchh", "pytorchj", "pytorchv", "PyTorchy",
    "pytorcm", "pytorcu", "PyTordh", "pytorh", "pytorhc", "pytorqh", "pytorrch", "pytprch", "pytroce",
    "pytroch", "PyTrosh", "pytspeak", "pytthon-binance", "pyttpmodule", "pytttsx3", "pytuon", "pytyon",
    "pytypier", "pytz3-dev", "pyurllib", "pyvrypto", "pywarder", "pywcc", "pywcontrol", "pywe", "pywgame",
    "pywhool", "pywin31", "pywin33", "pywint", "pywkill", "pywlgtb", "pywolle", "pywool", "pywosint",
    "pywpep", "pywpong", "pywpush", "pywstr", "pywstudy", "pywz", "pyxrypto", "pyyalm", "pyyaml-selenium",
    "pyycodestyle", "pyygame", "pyyinstaller", "PyYMAL", "pyysocks", "pyython-binance", "pyytorch", "pyzelf",
    "pzgame", "PzTorch", "QakePip", "qaqaqazzz", "qaxtest-httpx", "qsteemp", "qtcolor", "quantiumbase",
    "quarejma-botnet", "quarejma-botnetx", "quarejma-erdem", "quasarlib", "quick-telegram-sender",
    "quick_telegram_sender", "quickwebbasicauth", "quistik", "quixstreaming", "qyrm-pipinject4", "qyrm_pipinject4",
    "r3quests", "r4quests", "rabin-sharmakobau", "ramcc", "ramcontrol", "ramhacked", "ramint", "ramkill",
    "rammask", "rampush", "ramreplace", "ramstr", "ramstudy", "ramtool", "randgame", "randgenlib", "randget",
    "randgui", "randintel", "randkill", "randlgtb", "randmc", "randmine", "randomad", "randomgame",
    "randomgrand", "randomhttp", "randomhydra", "randomized", "randomlgtb", "randompep", "randompip",
    "randomrandom", "randomsplit", "randomultra", "randomvisa", "randpaypal", "randpush", "randrange",
    "randsplit", "randtool", "raw-tool", "rawrequest", "rblxtools", "rbxtool", "rbxtools", "rcyptocompare",
    "rcyptofeed", "rdquests", "readycharz", "readycher", "reallydonothing", "realtek", "reaquests", "reauests",
    "redist", "reduests", "reencode", "reequests", "reequests-toolbelt", "reeuests", "regexparam", "relefpots",
    "releycrypt", "rensoflow", "renvidia", "reols", "replacead", "replacecpu", "replaceintel", "replacepy",
    "replacerand", "replacestr", "replacesuper", "reportgenpub", "reproof", "repull", "req-flask",
    "req-matplotlib", "req-os", "req-tools", "req7ests", "req8ests", "req_flask", "reqargs", "reqeist",
    "reqeosts", "reqests-toolbelt", "reqeuste", "reqeusts-toolbelt", "reqeustx", "reqeustz", "reqeyst",
    "reqhests", "reqinstaller", "reqirements", "reqiremnets", "reqiremnts", "reqiurements", "reqiurementstxt",
    "reqiuremnets", "reqjuests", "reqkests", "reqoests", "reqquest", "reqquests", "reqquests-toolbelt",
    "reqsests", "reqsystem", "reqtik", "reqtrade", "requ-sts", "requas", "requeests", "requeests-toolbelt",
    "requeits", "requeksts", "requekts", "requeqsts", "reques-new-2022", "requesfs", "requesgt", "requesks",
    "requesqs", "requesrs", "requesrts", "requess", "requess-toolbelt", "requesst-toolbelt", "requessts",
    "requessts-toolbelt", "request", "request-get", "request-supporter", "request-toolbelt", "requesta",
    "requeste", "requesting", "requestlib", "requestlogger", "requestn", "requestr", "requests-beta",
    "requests-flask", "requests-http", "requests-httpx", "requests-new-module", "requests-oolbelt",
    "requests-otolbelt", "requests-pandas", "requests-sessions", "requests-tolbelt", "requests-tolobelt",
    "requests-toobelt", "requests-tooblelt", "requests-toolbbelt", "requests-toolbeelt", "requests-toolbel",
    "requests-toolbellt", "requests-toolbelt-v2", "requests-toolbeltt", "requests-toolbet", "requests-toolbetl",
    "requests-toolblet", "requests-toolblt", "requests-tooleblt", "requests-toolelt", "requests-toollbelt",
    "requests-tooolbelt", "requests-ttoolbelt", "requests-upgrade", "requests_darwin_lite", "requestspro",
    "requestss-toolbelt", "requesttest3", "requestts", "requestts-toolbelt", "requestw", "requesuts",
    "requesxs", "requesxt", "requesxts", "requesys", "requet", "requets-toolbelt", "requetsa", "requetsq",
    "requetss-toolbelt", "requetsts", "requewsts", "requfsts", "requiements", "requierement", "requierments",
    "requiest", "requiirements", "requiirementstx", "requiirementstxt", "requiirementsxt", "requiiremments",
    "requiiremnts", "requiirments", "requiremants", "requiremeents", "requiremenstx", "requiremenstxt",
    "requirementss", "requirementst", "requirementstt", "requirementsttx", "requirementstx", "requirementstxtt",
    "requirementstxtx", "requirementstxtxt", "requirementstxx", "requirementstxxt", "requirementt",
    "requirementtsxt", "requirementxstxt", "requirementxt", "requirementxtt", "requirementxxt", "requiremetns",
    "requiremetnstxt", "requiremetstx", "requiremetstxt", "requiremments", "requiremmentstxt", "requiremmentxt",
    "requiremmentxtxt", "requiremnets", "requiremnetstxt", "requiremnetxtxt", "requiremnts", "requiremntstx",
    "requiremntstxt", "requiremntxtxt", "requiremtns", "requirmeents", "requirment", "requirments",
    "requirmentss", "requirmentstx", "requirmentstxt", "requirmentstxtt", "requirrementstxt", "requirtements",
    "requists", "requiurement", "requiurementstxt", "requksts", "requnests", "requrementstxt", "requrest",
    "requriements", "requriments", "requset", "requsets", "requsets-toolbelt", "requssts", "requst",
    "requstes", "requsts-toolbelt", "requstss", "requstsss", "requstsssq", "requuests", "requuests-toolbelt",
    "requxsts", "requyests", "requzsts", "requеsts", "reqwestss", "reqzests", "rerandom", "resplit",
    "ressend", "ressy", "ressyy", "resuests", "reuests", "reuests-toolbelt", "reuirements", "reuqests-toolbelt",
    "reuquests", "reverse-shell", "reverse_shell", "ReverseShell", "revisa", "rewuests", "rfeqtrade",
    "rfquests", "rgbcolor", "rgbcolour", "rgparse", "rhermann", "rhermann-ct", "rhermann-sds", "rhermann-sdsm",
    "richcolor", "ripe-atlas-dyndns", "ripe.atlas.dyndns", "rllib3", "rmrentryutils", "ro-py-wrapper",
    "roblcx-cookie3", "roblcx_cookie3", "roblopython", "roblox-mod", "roblox-py-wrapper", "roblox-py1000",
    "roblox-python-wrapper", "roblox_mod", "roblox_py1000", "robloxapiaccess", "robloxfollowers", "robloxlogger",
    "robloxmod", "robloxpinreader", "robloxpinreaderr", "robloxpy-advanced", "robloxpython", "robonotif",
    "robux", "rock51", "rogressbar2", "rompt-toolkit", "ropython", "Rozenitro", "rpogressbar2", "rpompt-toolkit",
    "rq-websites", "rqeuests", "rqeuests-toolbelt", "rqrqrq", "rquest", "rquests-toolbelt", "rrequests",
    "rrequests-toolbelt", "rrlzjulvyx", "rrquests", "ruequest3", "rullib3", "rushbruhmalware", "rwquests",
    "ryptocompare", "ryptofeed", "s3tranfer", "s3transfere", "s3transferr", "s3trnasfer", "s3trnasfers",
    "saazsdz", "saazszxcdz", "sadsacxz", "safepackage", "Sagepay", "sagetesteight", "sagetestfive",
    "sagetestfour", "sagetestseven", "sagetestsix", "sagetestthree", "sagetesttwo", "salami3",
    "sanic-prometheus-qubit", "saturnian", "sc-concurrent-log-handler", "sc-oauth2", "scappy", "scarpy",
    "scavenger-py", "sccikit-learn", "sccrapy", "schubismomv2", "schubismomv3", "sciikit-learn", "sciikt-learn",
    "sciit-learn", "scikiit-learn", "scikit-earn", "scikit-elarn", "scikit-laern", "scikit-larn",
    "scikit-lean", "scikit-leanr", "scikit-lear", "scikit-learn-matplotlib", "scikit-learrn", "scikit-leearn",
    "scikit-leran", "scikit-llearn", "scikkit-learn", "scikt-learn", "scikti-learn", "sckiit-learn",
    "sckit-learn", "sclighte", "sclite", "scraapy", "scrapeasy-jkhjasdsad", "scrapeasy-juhsdgfjs",
    "scrappers", "scrappers-dev", "scrapyy", "scray", "screencaptned", "screenshiterr", "scrpay", "scrrapy",
    "sdgsdghsdhd1", "sdgsdghsdhd12", "sdk-cli-v2", "sdk-cli-v2-public", "sea-django-mysqlpool", "seabron",
    "secbg", "seccache", "secretslib", "secrevthree", "secrevtwo", "security-util", "seelenium", "seelnium",
    "seleenim", "seleenimu", "seleenium", "seleeniumm", "seleinium", "seleiniumm", "seleinuim", "seleinum",
    "seleium", "seleiumm", "selemiumm", "selemni", "selemnim", "selemnium", "selemniumm", "seleneium",
    "seleniium", "selenimn", "seleniu", "selenium-matplotlib", "seleniumm", "seleniumunclickable", "seleniun",
    "seleniuum", "selennim", "selennium", "selenniumm", "selennuim", "selenuim", "selenuimm", "selenum",
    "selenyum", "seleunium", "selfadcpu", "selfadintellgtb", "selfadmaskmask", "selfadpullint", "selfbot-api22",
    "selfcandyed", "selfcandyhttpnvidia", "selfcandykillurl", "selfcandyrand", "selfcandystudystudy",
    "selfccgame", "selfcckillad", "selfccpeppep", "selfccping", "selfccpullint", "selfccpyw", "selfccreplacever",
    "selfccurlpong", "selfccvirtualgame", "selfcontroledintel", "selfcontroledreplace", "selfcontrolhttppip",
    "selfcontrolpy", "selfcontrolrandom", "selfcontrolstr", "selfcontrolurlencode", "selfcpugrand",
    "selfcpuinfo", "selfcpuintelpip", "selfcpuintgame", "selfcpuintsplit", "selfcpuloadnvidia",
    "selfcpupingcc", "selfcpupongkill", "selfcpusplit", "selfcraftcontrolload", "selfcraftint",
    "selfcraftsuperhacked", "selfcraftvmultra", "selfcvcvmask", "selfcvcvsuper", "selfcvedpep", "selfcvhttp",
    "selfcvinfoad", "selfcvinfourl", "selfcvmc", "selfcvpaypalram", "selfcvpull", "selfcvpushpull",
    "selfcvstrpong", "selfcvvirtual", "selfededpy", "selfedgamestudy", "selfedintsuper", "selfednvidiatool",
    "selfedrandomrandom", "selfedrerand", "selfencodecandy", "selfencodeload", "selfencodemaskpong",
    "selfencodenvidiaurl", "selfencodepostmc", "selfencodesplit", "selfgamecvultra", "selfgameencode",
    "selfgamepost", "selfgamepullad", "selfgamepushstr", "selfgamepypost", "selfgamesplit", "selfgametool",
    "selfgamevirtualad", "selfgetgrand", "selfgetmask", "selfgetminecandy", "selfgetstudyram",
    "selfgetultrapong", "selfgrandccintel", "selfgrandedgame", "selfgrandgametool", "selfgrandstrpyw",
    "selfguiad", "selfguiccproof", "selfguilibrand", "selfguiloadget", "selfguiurlpush", "selfhackedccmc",
    "selfhackedccpull", "selfhackedcpuvirtual", "selfhackedencode", "selfhackedlib", "selfhackednvidiamc",
    "selfhackedramcc", "selfhackedrandomstudy", "selfhackedrandvirtual", "selfhackedrepong", "selfhttpcraftultra",
    "selfhttpinfo", "selfhttpmcpush", "selfhttppep", "selfhttppy", "selfhttppyw", "selfhttpsplited",
    "selfhttpstring", "selfhttpvmcontrol", "selfhydrainfoproof", "selfhydrakill", "selfhydramask",
    "selfhydrapongpaypal", "selfhydrastudycc", "selfinfocraftlib", "selfinfoencode", "selfinfoencodegame",
    "selfinfointpy", "selfinfolgtbhydra", "selfinfopaypal", "selfinfopaypalad", "selfinfoponged",
    "selfinfopushpip", "selfinforand", "selfinforandom", "selfinfovisakill", "selfintcontrol", "selfintcv",
    "selfintelcpu", "selfinteledreplace", "selfintelpullpost", "selfintelsplitlgtb", "selfintmaskreplace",
    "selfintstringpong", "selfintstringstr", "selfintsuper", "selfinturlstudy", "selfintvisa", "selfkillcontrol",
    "selfkillgameintel", "selfkillguipaypal", "selfkilllgtb", "selfkillpinghydra", "selfkillsplitpy",
    "selfkillvirtualhydra", "selflgtbcc", "selflgtbcontrolpyw", "selflgtbpaypal", "selflgtbpep",
    "selflgtbpostrand", "selflgtbproofgrand", "selflgtbstringload", "selflibcv", "selflibencodetool",
    "selflibmineload", "selflibreplace", "selfloadcpu", "selfloadcvencode", "selfloadcvhttp", "selfloadedpy",
    "selfloadpywnvidia", "selfloadtoolcc", "selfmaskcandyrandom", "selfmaskcpu", "selfmaskgamepull",
    "selfmaskhydravm", "selfmaskminehacked", "selfmasksuper", "selfmaskultra", "selfmaskvisa", "selfmccontrolstudy",
    "selfmcosintrandom", "selfmcpipcpu", "selfmcvisapy", "selfmineencodestr", "selfminegrandkill",
    "selfminemasked", "selfminemasknvidia", "selfminepywurl", "selfmineultraram", "selfnvidiaccpong",
    "selfnvidiareplaceload", "selfosintcraftlib", "selfosintgame", "selfosintget", "selfosintgrandrandom",
    "selfosintlgtbstr", "selfosintlib", "selfosintping", "selfosintpostlib", "selfosintpywmask",
    "selfosintsplithydra", "selfosintstringcpu", "selfosintultrahacked", "selfosinturlre", "selfosintver",
    "selfpaypaladvm", "selfpaypalcontrolsuper", "selfpaypalloadpaypal", "selfpaypalpushgame", "selfpaypalram",
    "selfpaypalstr", "selfpaypalvm", "selfpepad", "selfpepintel", "selfpeppippull", "selfpeppongload",
    "selfpeppullram", "selfpepreplaceload", "selfpepstudylib", "selfpepultraad", "selfpepvirtual",
    "selfpingintelvm", "selfpingponggame", "selfpingram", "selfpingrampep", "selfpingultragame", "selfpingvisa",
    "selfpipmask", "selfpippaypalpyw", "selfpippongpip", "selfpippyreplace", "selfpongcpu", "selfpongencode",
    "selfpongencodeintel", "selfponggui", "selfpongnvidia", "selfpongpull", "selfpongurl", "selfpostcontrol",
    "selfpostguihttp", "selfposthydra", "selfpostmc", "selfpostmcintel", "selfpostponghydra", "selfpostramstring",
    "selfpoststringvm", "selfpostver", "selfproofgame", "selfproofint", "selfproofintelosint", "selfproofkillcraft",
    "selfproofloadram", "selfproofnvidiagrand", "selfproofpipmine", "selfproofpullsplit", "selfproofstudyrand",
    "selfproofurl", "selfpullint", "selfpullsplitcraft", "selfpullstringpong", "selfpullstringtool",
    "selfpullstringver", "selfpushcv", "selfpushlgtbnvidia", "selfpyintpaypal", "selfpyosinthacked",
    "selfpyosintmine", "selfpypeppush", "selfpypywpong", "selfpyver", "selfpyverint", "selfpywgame",
    "selfpywgamesuper", "selfpywgetget", "selfpywhydragui", "selfpywkillgui", "selfpywloadmine",
    "selfpywloadpong", "selfpywosint", "selfpywpaypalultra", "selfpywreplace", "selframintelosint", "selframlgtb",
    "selframproof", "selframstring", "selframstudyget", "selfrandcraftreplace", "selfrandloadcv",
    "selfrandomcandypy", "selfrandomhackedstudy", "selfrandomintosint", "selfrandompep", "selfrandompiposint",
    "selfrandompostinfo", "selfrandompostint", "selfrandompullver", "selfrandomram", "selfrandomstrad",
    "selfrandreplacerand", "selfrecandyping", "selfrecpu", "selfrehttp", "selfrekillrand", "selfrepaypalhttp",
    "selfreplaceencodepush", "selfreplacelib", "selfreplacepong", "selfreplacereplace", "selfreplacesplit",
    "selfreplacetoolreplace", "selfreplacevisa", "selfsplitlib", "selfsplitpush", "selfsplitrandompyw",
    "selfsplitreplacecraft", "selfsplitultra", "selfstringkill", "selfstringlib", "selfstringpullpush",
    "selfstrkill", "selfstrpyw", "selfstrresuper", "selfstrultrapush", "selfstudycontrolpush", "selfstudygrand",
    "selfstudyhydra", "selfstudyhydrakill", "selfstudyintel", "selfstudyintelcc", "selfstudyintstudy",
    "selfstudynvidiapep", "selfstudytool", "selfstudytoolrandom", "selfsuperaded", "selfsupercandy",
    "selfsupered", "selfsupergameencode", "selfsuperpyw", "selfsuperstrpaypal", "selfsupervisa",
    "selftoolguihttp", "selftoolinfoint", "selftoolinfover", "selftoollibnvidia", "selftoolmask",
    "selftoolpyint", "selftoolreplace", "selftoolvirtualpep", "selfultracraft", "selfultraencodevm",
    "selfultrahydra", "selfultrainfocraft", "selfultraintelvisa", "selfultramc", "selfultrapaypalencode",
    "selfultrapingpush", "selfultrapyw", "selfultraver", "selfultravirtual", "selfurlcpu", "selfurlcraft",
    "selfurlminecontrol", "selfurlrandomver", "selfurlultra", "selfverccint", "selfvergame", "selfvergrandstr",
    "selfvergui", "selfverkill", "selfvermaskpush", "selfverpushget", "selfverrand", "selfvirtualencode",
    "selfvirtualhackedhttp", "selfvirtualhydra", "selfvirtualinfoencode", "selfvirtualloadrand",
    "selfvirtualreosint", "selfvisacontrolcraft", "selfvisacpu", "selfvisagrandurl", "selfvisaosint",
    "selfvisaosintreplace", "selfvisapaypalmine", "selfvisapostosint", "selfvisarandompush", "selfvisareplacemc",
    "selfvisaurlpong", "selfvisavmgame", "selfvmad", "selfvmcraftpush", "selfvmedram", "selfvmload",
    "selfvmproofgui", "selfvmreplace", "selib", "seliniumm", "seliniumn", "selinum", "selleium", "selleniium",
    "sellenim", "sellenium", "selleniumm", "sellinium", "selneium", "selnium", "selunium", "semdb", "semdber",
    "semurgdb", "sentinelone", "sentinelone-sdk", "sentinelonesdk", "servantcord", "setdotwork", "setnetwork",
    "settinginmaass", "setup1nter", "setupint3", "setupint3s", "setupynts", "sfox-ecdsa", "sgmm", "shaasigma",
    "shaders", "shaderz", "shadw", "shellapp", "shellapp1", "shellexec", "shenghuo2-getshell",
    "shenghuo2_getshell", "sherm", "shermed", "shermly", "sherms", "shikitesting", "shimi", "shiwers", "shom",
    "siamphishtest", "siamrahman", "siamttviews", "sickit-learn", "siimplejson", "sijplejso", "Sijplejson",
    "sikit-learn", "Simepljson", "simlejson", "simlpejson", "simmplejson", "Simolejson", "simpejso",
    "simpejson", "simpeljson", "simpjson", "Simpkejson", "SimpleCalc-2022", "simpleejson", "Simplejason",
    "Simplejdon", "simplejjson", "simplejon", "simplejosn", "simplejsn", "simplejsno", "simplejso",
    "Simplejsoh", "Simplejsoj", "simplejsonn", "simplejsoon", "simplejsson", "simplesjon", "simpleson",
    "simpljeson", "simpljson", "simpllejson", "simplyjson", "Simpoejson", "simpplejson", "sintaxisoyyo",
    "Siplejason", "siplejson", "sipmlejson", "sitechdemo100", "Sjimplejson", "Sjmplejson", "skd64", "slana",
    "sleenium", "slenium", "slintscreen", "sln1550hello", "sloana", "sluheczo", "smb", "smiplejson", "snelt",
    "snwproxies", "soalna", "soana", "sobit-ishlar", "sobit_ishlar", "social-checker", "social-scrapper",
    "social-scrappers", "social_scrapper", "social_scrappers", "socksproxies", "solaa", "solaan", "solaana",
    "solanaa", "solanna", "sollana", "solna", "solnaa", "soolana", "soulnitro", "soup2", "soupcsstools",
    "souptools", "spacefilterapi", "spacestudio-orbit-propagation", "SpammingSynonym", "SpammingSynonyms",
    "spamysynonym", "sparkk", "sparklog", "speech-dtw", "speechrecongition", "sphinx-rtd-theme-cilium",
    "splib-http", "splitcc", "splithydra", "splitintel", "splitmask", "splitnvidia", "splitpep", "splitram",
    "splitsplit", "splitstudy", "splitver", "spookyimagelogger", "spotify2youtubemusic", "spyme", "SpyWare",
    "sql-to-sqlite", "sqlachemy", "sqlalcemy", "sqlalchemy-install", "sqlalchemy-os", "sqlalchemy-presto",
    "sqlalchemy-requests", "sqlalchemy_os", "srcapy", "srv-configs", "ss-concurrent-log-handler",
    "ssc-concurrent-log-handler", "sscikit-learn", "sscrapy", "sselenium", "ssimplejson", "ssolana",
    "stackstorm-runner-action-chain", "startup-entrypoints", "statmodel", "statmodels", "statsmodel",
    "stealthpy", "steembase", "stillrequestsa", "strbeautify", "strcolored", "strcolrify", "strgrand",
    "strhydra", "strinfer", "strinfo", "stringcc", "stringe", "stringencode", "stringgrand", "stringintel",
    "stringrand", "stringultra", "stripe-client", "stripe-client-py", "stripe.client-py", "stripepy",
    "strload", "strpep", "strpull", "strram", "strrandom", "strreplace", "strstyle", "strstylle", "strsuper",
    "studycc", "studyhydra", "studylib", "studyload", "studymine", "studypong", "studyrandom", "studystr",
    "studyver", "styling", "sudo2", "suffer", "supergrand", "superint", "superpull", "superreplace",
    "supertest9188", "support-dev", "support-hub", "support_dev", "support_hub", "supra-style", "supra_style",
    "sutiltype", "swapmempool", "swiftypy", "sylex-syntax", "sylexnaranjoo", "syns-knox-xss-allwhere",
    "syns_knox_xss_allwhere", "syntax-init", "syntaxcode", "synthetictest", "synthetictest1", "sys-scikit-learn",
    "sys-selenium", "sysargvsox", "syscoloringextensionv2", "syscoloringspkg", "syscoloringspkgs", "syscolorv2",
    "syscolouringkitsv2", "syscolouringsextv1", "syscolourkitsv2", "syscolourtoolkit", "syscord",
    "syscryptlibv2", "syscryptographymodsv2", "sysdatalib", "sysfontingpkgv1", "syslog-ng", "syslog-udp",
    "syssqlite2package", "syssqlite2toolsv2", "syssqlite2toolv2", "syssqlite3liberyv1", "syssqlite3v2",
    "syssqliteaddv2", "syssqlitedbextension", "syssqlitedbpackagev1", "syssqlitelibery", "syssqlitemods",
    "systemdemon", "sysuptoer", "tablediter", "tabulation", "tcalendar", "tckalendar", "tdwtauthauthentication",
    "teensorflow", "telerer", "telthi", "temsorflow", "tennsorflow", "tenorflow", "tenosrflow", "tensnflow",
    "tensobflow", "tensofklow", "tensofl9w", "tensofla", "tensoflaow", "tensofleow", "tensofliw", "tensofllow",
    "tensofloaw", "tensoflod", "tensoflolw", "tensoflom", "tensoflomw", "tensoflonw", "tensoflor",
    "tensoflouw", "tensoflpw", "tensoflqw", "tensoflsw", "tensoflw", "tensoflxow", "tensofpow", "tensofrlow",
    "tensogflow", "tensoorflow", "tensorfflow", "tensorfllow", "tensorflo", "tensorfloow", "tensorfloww",
    "tensorflw", "tensorflwo", "tensorfolw", "tensorlfow", "tensorlow", "tensorrflow", "tensourflow",
    "tensrflow", "tensrflwo", "tensroflow", "tenssorflow", "tensxoflow", "tensxxfxxk", "tequests",
    "termcolour", "tesla-faas2", "tesnorflow", "tesoaoerm", "tesorflow", "tessssssssss", "test-23234231",
    "test-async", "test-test-test123", "test23414234234", "test24234", "test_23234231", "testbrojct2",
    "testdontdownloadthis", "testdufou", "testedwin", "testepassword-generate", "testfiwldsd21233s",
    "testing1232", "testingbrooasqa", "testinghelloma", "testingiasdf1", "testingiavv", "testingijijwdaijdwa",
    "testingvvv3xx", "testiramtikurbu", "testkaralpoc45654", "testlibs111", "testlibtaha", "testnetdubled",
    "testomadaoto", "testpackage1mcpe", "testpackages159", "testpipper", "testpipperz", "testpipxyz3",
    "testpipxyz4", "testpkg3322", "testwhitesnake", "testwhitesnake123a", "testwhitesnakemodule", "tfit",
    "tg-bulk-sender", "tg_bulk_sender", "tgrequest", "tgrequests", "thebypasstool", "theerum", "thesar",
    "thesis-uniud-package", "thesis_package", "threadin", "threadxpools", "threeding", "tiktok-bots",
    "tiktok-filter-api", "tiktok-phone-cheker", "tiktok4", "tiktok5", "tiktok6", "tiktok8", "tiktok9",
    "tiktok_phone_cheker", "tiktokthon", "tiny423", "tiny43", "tiny433", "tinyad", "tinyad1", "tinyad2",
    "tjajsd", "tkaclendar", "tkalendar", "tkcaalendar", "tkcaelndar", "tkcaendar", "tkcalednar", "tkcalenadr",
    "tkcalenda", "tkcalendaar", "tkcalendarr", "tkcalenddar", "tkcalendr", "tkcalendra", "tkcalenndar",
    "tkcallendar", "tkcalndar", "tkcalnedar", "tkccalendar", "tkclaendar", "tkclendar", "tkinter-message-box",
    "tkintres", "tkintrs", "tkkcalendar", "tkniter", "tls-bypass", "tls-client-py", "tls-python",
    "tls_client-py", "tlsclient-api", "tlsproxies", "tnesorflow", "tnsorflow", "tokenpath", "tommygtst",
    "tomproxies", "toolcv", "tooled", "toolgrand", "toolhydra", "toollgtb", "toolload", "toolmine",
    "toolnvidia", "toolpip", "toolproof", "toolpull", "toolrand", "toolstudy", "toolvisa", "tpadpip",
    "tpadvmpip", "tpcandycontrolpush", "tpcandycontrolrand", "tpcandykill", "tpcandykillget", "tpcandyloadvisa",
    "tpcandyminead", "tpcandypong", "tpcandyreed", "tpccguipip", "tpccintelcontrol", "tpccpinghacked",
    "tpcontrolint", "tpcontrolkillrandom", "tpcontrolpywreplace", "tpcontrolsplit", "tpcontrolstr",
    "tpcontrolvmpost", "tpcpuedosint", "tpcpuintpep", "tpcpupingreplace", "tpcpuramhttp", "tpcpustrvm",
    "tpcraftad", "tpcraftcandy", "tpcraftcandypyw", "tpcraftcraftencode", "tpcrafthydracandy",
    "tpcraftvisasuper", "tpcvadlib", "tpcvgrandhacked", "tpcvinfo", "tpcvnvidiasuper", "tpcvpygui",
    "tpcvre", "tpcvultrainfo", "tpcvurlpong", "tpcvverencode", "tpedcpulib", "tpedhydrapush", "tpedinfo",
    "tpedintload", "tpednvidiakill", "tpedpingpull", "tpedramver", "tpedstrproof", "tpedultraintel",
    "tpencodecandyosint", "tpencodeinteled", "tpencodereplacepush", "tpgamegetnvidia", "tpgamehacked",
    "tpgamelibhacked", "tpgamemine", "tpgamepip", "tpgamestr", "tpgetadstr", "tpgethydra", "tpgetlibget",
    "tpgetosint", "tpgetpipcv", "tpgrandcontrol", "tpgrandencodegui", "tpgrandencodever", "tpgrandkillhydra",
    "tpgrandlibload", "tpgrandmask", "tpgrandvmpaypal", "tpguicontrolcv", "tpguied", "tpguigrandreplace",
    "tpguiintel", "tpguiintelcv", "tpguiosintre", "tpguistring", "tpguiverpyw", "tphackedad",
    "tphackedcontroled", "tphackedcraftnvidia", "tphackeded", "tphackedhttpultra", "tphackedhydraproof",
    "tphackedintelhydra", "tphackedrandhydra", "tphackedsplitcraft", "tphttpLGTBrandom", "tphttpsuperpep",
    "tphydracvmask", "tphydraencode", "tphydraloadsuper", "tphydramask", "tphydrapushstr", "tphydratoolsplit",
    "tpinfoencodeget", "tpinforandcpu", "tpinfourllgtb", "tpinfovirtualre", "tpintelcpuver", "tpintelgetlib",
    "tpintelgrandlgtb", "tpintelpullcpu", "tpintelramhacked", "tpintelrandom", "tpintelvercandy",
    "tpintgamevirtual", "tpintgrandre", "tpintinfohacked", "tpintpaypal", "tpintver", "tpkillcandykill",
    "tpkillkillsplit", "tpkillmask", "tpkillosintcontrol", "tpkillpeppep", "tpkillproof", "tpkillproofpip",
    "tpkilltool", "tplgtbcontrol", "tplgtbpull", "tplibhydrahacked", "tplibhydrainfo", "tplibinfo",
    "tplibpullpyw", "tpliburlver", "tplibvirtualram", "tploadget", "tploadgrand", "tploadgui", "tploadkillping",
    "tploadloadosint", "tploadpep", "tploadpostosint", "tploadpulllgtb", "tploadreplace", "tploadreplacestring",
    "tpmaskcontrolhydra", "tpmaskintelpull", "tpmasklibsplit", "tpmaskmcpush", "tpmaskpaypalping",
    "tpmaskstudy", "tpmaskvisacontrol", "tpmccvpong", "tpmcpongpaypal", "tpmcrandgrand", "tpmcverhacked",
    "tpmcvirtual", "tpmcvisa", "tpminegetintel", "tpminehackedsuper", "tpminekill", "tpmineloadtool",
    "tpminesuperpip", "tpminever", "tpnvidiacv", "tpnvidiaintelpush", "tpnvidiapong", "tpnvidiarandsplit",
    "tpnvidiareurl", "tpnvidiasplit", "tposintadget", "tposintcraftram", "tposinthackedstring", "tposintpywcraft",
    "tposintvirtual", "tppaypaledkill", "tppaypalmine", "tppaypalpytool", "tppaypalrandomurl", "tppaypalsuper",
    "tppaypalver", "tppepcontrolstring", "tppeppingram", "tppeppushpyw", "tppepsplit", "tppepultra",
    "tppingcpuget", "tppinghackedpaypal", "tppinghydrainfo", "tppingsplitcc", "tppingstring", "tppingultraload",
    "tppingvercv", "tppingvirtualrand", "tppipgrand", "tppiphttpstring", "tppiprandhydra", "tppipvm",
    "tppongadultra", "tpponglgtbre", "tpponglib", "tppongpy", "tppongsplitrandom", "tppongvisapip",
    "tppostcandynvidia", "tppostedping", "tppostsplit", "tppoststringcontrol", "tppoststrsuper",
    "tpproofgameed", "tpproofkill", "tpproofmcultra", "tpproofpip", "tpprooftool", "tpproofultra", "tppullcv",
    "tppullmcreplace", "tppullpush", "tppullpy", "tppullpyad", "tppushcontrol", "tppushcpucandy",
    "tppushmaskpull", "tppushvmgrand", "tppyhttprand", "tppyinfo", "tppyintgame", "tppylgtbnvidia",
    "tppylgtbreplace", "tppyosintrand", "tppypip", "tppypiplgtb", "tppyproof", "tppyrandomed", "tppyreplacestr",
    "tppywget", "tppywgetload", "tppywintelgame", "tppywloadhydra", "tppywmc", "tppywpostram", "tppywproofcpu",
    "tppywproofstr", "tppywpullstr", "tppywrandompull", "tppywstudy", "tppywultraint", "tpramgrandcontrol",
    "tpramhydracpu", "tpramosint", "tpramosintpost", "tprampingload", "tprampipre", "tpramtoolpong",
    "tpramvisa", "tprandcandy", "tprandguilib", "tprandomosint", "tprandompullpull", "tprandomultravisa",
    "tprandping", "tprandpull", "tprandpushvirtual", "tprandrehttp", "tprandstringpep", "tprecandy",
    "tpreccmine", "tprelibcpu", "tpreloadad", "tpremineosint", "tprepaypalpull", "tpreplacecraft",
    "tpreplacecrafthacked", "tpreplacegrandrandom", "tpreplacekillram", "tpreplacelib", "tpreplacepush",
    "tpreproof", "tprepullpush", "tpreurlultra", "tpsplitgui", "tpsplitlgtb", "tpsplitlibpaypal",
    "tpsplitloadhydra", "tpsplitpostpong", "tpsplitpushpush", "tpsplitram", "tpsplitrandcv", "tpsplitstudy",
    "tpstrcpu", "tpstrgethydra", "tpstringcraftget", "tpstringhackedstudy", "tpstringintpong", "tpstringlgtb",
    "tpstringmask", "tpstringpep", "tpstringverstudy", "tpstrintelurl", "tpstrlibhydra", "tpstrmask",
    "tpstrmaskpy", "tpstrmcint", "tpstrpepLGTB", "tpstrpush", "tpstrrandomint", "tpstrsplitvirtual",
    "tpstudygetreplace", "tpstudyosintvirtual", "tpstudypingrand", "tpstudyproofhttp", "tpstudystringpep",
    "tpstudyver", "tpstudyvirtual", "tpsupermcmc", "tpsuperosintmask", "tpsuperpyw", "tpsuperultramc",
    "tptoolcccpu", "tptoolcpusplit", "tptoolintelram", "tptoolkill", "tptoolmask", "tptoolpywgui", "tpultraencode",
    "tpultragui", "tpultrainfo", "tpultraintel", "tpultralgtbcandy", "tpultrapullrand", "tpultravmpy",
    "tpurladpy", "tpurlguiint", "tpurlintel", "tpurllib", "tpurlpaypalcraft", "tpurlpullpaypal", "tpvercontrol",
    "tpverencodeinfo", "tpverguipost", "tpvervmsplit", "tpvirtualcontrolcraft", "tpvirtualcontrolmc",
    "tpvirtualhttp", "tpvirtualpush", "tpvirtualpywproof", "tpvirtualrandom", "tpvirtualurl", "tpvisaguipost",
    "tpvisalgtbcv", "tpvisaosint", "tpvisapyw", "tpvisaram", "tpvisasplitram", "tpvmcpustudy", "tpvmcraftmine",
    "tpvmencodecv", "tpvmpipsplit", "tpvmpy", "tpvmstringgrand", "tqmd", "trc20-unlocker", "trejklfffffffffgdjg",
    "trexcolors", "trnsorflow", "trustpiphuh", "trustpiphuh1", "trustypip", "tryconf", "tryhackme-offensive",
    "tryhackme_offensive", "tryquests", "tshawn-wrce", "ttensorflow-gpu", "ttkcalendar", "ttlo", "tucan-x",
    "twittter", "twyne", "type a own package name", "type-a-own-package-name", "type-color", "types-for-adobe",
    "typestring", "typesutil", "typing-extnesions", "u283udsfru", "ucap", "uihuihiuhiuhiu", "ulibasset",
    "ulrlib3", "ultracc", "ultracontrol", "ultracv", "ultralib", "ultramc", "ultraobfuscator", "ultraproof",
    "ultrarand", "ultrarequests", "uniswap-math", "update-mss", "update-request", "update-requests",
    "updated-requests", "updater12", "updlibupload", "upggrade-requests", "upgini", "upgrade-requests",
    "upgrade-requestss", "upgrade-requestt", "urelib3", "urklib3", "url-requests", "urlib3", "urlkib3",
    "urllb", "urllb3", "urllbi3", "urllgtb", "urlli3", "urlli3b", "urllib", "urllib12", "urllib33",
    "urllib3installer", "urllib3loader", "urllib7", "urllibb3", "urllibdownloader", "urllibinstaller",
    "urllibloader", "urlliib3", "urllitelib", "urlllib", "urlmine", "urlnvidia", "urlpaypal", "urlreplace",
    "urlsplit", "urlver", "urlvisa", "urolib3", "urrllib3", "urtelib32", "urz", "user-agents-parser ",
    "useragentclient", "utilitytools", "uurllib3", "uzzywuzzy", "v4pe", "validatekey", "value2", "value3",
    "valyrian-debug", "verad", "vercc", "verhttp", "verint", "vermillion", "vernvidia", "verosint", "verpong",
    "verpush", "verpy", "vertica_parser", "vervisa", "very-hackerman", "vibrant", "virtualcc", "virtualcontrol",
    "virtualhttp", "virtualhydra", "virtualnv", "virtualstudy", "virtualvirtual", "visahydra", "visakill",
    "visaload", "visamc", "visapull", "visapush", "visaram", "visastudy", "visatool", "visavirtual", "vmconnect",
    "vmed", "vmgame", "vmhttp", "vmnvidia", "vmstudy", "vmver", "vmvirtual", "vper", "vpyer", "vvyper",
    "vyepr", "vyer", "vype", "vypeer", "vyperr", "vypper", "vypre", "vyyper", "w3b", "w3b-py", "w3bt00n",
    "w3eb", "wadokwaokda", "warprnnt-pytorch", "wb3", "wb3-py", "wbe3", "wbe3-py", "wbesocket-client",
    "wbesockets", "wbsocket-client", "wdb3", "wdrags", "we3", "we3-py", "we3b", "Web-requests-autmoation",
    "web2", "web3-0py", "web3-essential", "web3-p6", "web3-p7", "web3-po", "web3-pu", "web3-py9", "web3-pyu",
    "web3-pyy", "web3e", "web3q", "web3toolz", "web3toolzfor", "web3txtools", "web4-py", "webbsocket-client",
    "webbsockets", "webhosting", "webocket-client", "webockets", "weboscket-client", "webosckets",
    "webquickauth", "webscket-client", "websckets", "webscoket-client", "webscokets", "websoccket-client",
    "websocckets", "websocekt-client", "websocet-client", "websocets", "websocke-client", "websockeet-client",
    "websockeets", "websockes", "websockest", "websocket-cclient", "websocket-cient", "websocket-cleint",
    "websocket-clieent", "websocket-clien", "websocket-cliennt", "websocket-clientt", "websocket-cliet",
    "websocket-clietn", "websocket-clinet", "websocket-clint", "websocket-cllient", "websocket-lcient",
    "websocket-lient", "websocketss", "websockett-client", "websocketts", "websockket-client", "websockkets",
    "websockt-client", "websockte-client", "websocktes", "websockts", "websokcet-client", "websokcets",
    "websoket-client", "websokets", "websoocket-client", "websoockets", "webssocket-client", "webssockets",
    "webt00n", "webt3", "webtoongen", "webtraste", "webtungen", "wec3", "weeb3-py", "weebsocket-client",
    "weebsockets", "WeeCoder", "WeedyCoder", "weg3", "werb3", "wesbocket-client", "wesbockets", "wesocket-client",
    "wesockets", "wessycord", "wev3", "wev3-py", "wheel-cache", "wifitool", "win23crypt", "winrpcexploit",
    "WkquBsXEkbXn", "wmi-toolbox", "woodwhalehack114", "wue", "wweb", "wweb3", "wweb3-py", "wwebsocket-client",
    "wwebsockets", "wxpay-comm", "wxpayproto", "xamp", "xbox-authorization", "xbox-promo-checker-api",
    "xboxkeyauth", "xboxlivepy", "xboxredeemer", "xboxsolver", "xcryptography", "xFileSyncerx",
    "xgoogle-cloud-core", "xgoogle-cloud-storage", "xhttpsp", "xiedemo", "xlatency", "xlibrary", "xllsxwriter",
    "xlssxwriter", "xlswxriter", "xlsxriter", "xlsxrwiter", "xlsxwiter", "xlsxwrier", "xlsxwrietr",
    "xlsxwriiter", "xlsxwrite", "xlsxwriteer", "xlsxwriterr", "xlsxwritr", "xlsxwritre", "xlsxwritter",
    "xlsxwrriter", "xlsxwrter", "xlsxwrtier", "xlsxwwriter", "xlsxxwriter", "xlxswriter", "xlxwriter",
    "xmlbuilder3", "xoloaelvcsjwnt", "xoloaghvurilnh", "xoloaoqcjnreyc", "xoloarmmonwkmr", "xoloaydpctxiyp",
    "xoloayihnhlayp", "xolobbbccc", "xolobgcbdndabm", "xolobwritbrulv", "xolobzvfburelm", "xoloctwuaywkna",
    "xolocyawkmylds", "xolodevcceglww", "xolodqijhnjgte", "xolodvbqgrfohn", "xolodxhrsnrxai", "xolodyntlnewtp",
    "xoloeduccelifz", "xolofmdvxqvbmp", "xoloftiqwxxhje", "xolofucxlcmyke", "xologpbhyminnv", "xologrekjlqzxj",
    "xolohnetekcjdz", "xolojbxzzttwpk", "xolojgmnizxche", "xolojhzyppbsow", "xolokadyqehtbs", "xolokqhufyiwyq",
    "xolokvhcqvifyf", "xololcuakbzbuu", "xolomayflwnfmy", "xolomdabxhhrue", "xolomjqalvrpmp", "xolommyjlqlhsw",
    "xolonjucebiwfa", "xolookvryqetgd", "xolopfydnuxyfh", "xolopwjaansvnd", "xoloqgavocbfcd", "xoloqiyrnnqwll",
    "xoloqmotdjpbic", "xoloqvqexetcqo", "xoloqyrmkojrfm", "xolortpdcanegu", "xolosafhpodvqo", "xolosamsdyhcfa",
    "xolosbmgfnvgzi", "xolostfqwqiaxe", "xolosxelwsesnp", "xolosybevwfsny", "xolotabiamysla", "xolotcgstfiguu",
    "xolotopgrabber", "xolotxobrzatpu", "xoloulfkhiyywc", "xolouwdmgbgkvr", "xolovqryjphftd", "xolovzgfkdamoj",
    "xolowgdmsxvuwm", "xolowqffntthtb", "xoloxrxcxfywtm", "xoloxwmellxliq", "xoloxygjidhpoo", "xoloyfczocogra",
    "xoloytubfihhsa", "xolozamdgbxywf", "xolozpnyeyhirx", "xpip", "xsetuptools", "xss", "xsxwriter",
    "xxlsxwriter", "xxxsss", "xxxxssss", "yandex-map", "yaudio", "yautogui", "ycodestyle", "ycrypto",
    "ycryptodome", "yelp-cgeom1", "yessirmian", "yffinance", "yfiance", "yfiannce", "yfiinance", "yfinaance",
    "yfinace", "yfinacne", "yfinancce", "yfinancee", "yfinane", "yfinanec", "yfinannce", "yfinnace",
    "yfinnance", "yfinnce", "yfnance", "yfniance", "ygame", "yiffparty", "yinance", "yinstaller", "ymongo",
    "yocolor", "yohewoasaw", "youhans", "youtube-new", "ypcodestyle", "yper", "ypinstaller", "ypsocks",
    "ypthon-binance", "yptt", "ysocks", "ython-binance", "ytorch", "yvper", "ywin32", "yyfinance", "zafira",
    "zefkopzekfo", "zelixnitro", "zeubilamouche", "zhpt1cscoe", "zlibsrc", "znomig", "zoom-pyutils",
    "zorosnitro", "zproxy", "zproxy2","reqeusts","urllib3s", "base64io","internal-lib", "python3-dateutil",
    "ztasimb", "zuppa", "zwhrce", "zydnitro", "zyqnuutupjerllnbxaeq","company-utils"
}

SUSPICIOUS_KEYWORDS = [
    "stealer", "hack", "crack", "cheat", "trojan", "backdoor",
    "keylogger", "phish", "fake", "malicious", "evil", "suspicious", "suspious"
]

CRITICAL_PACKAGE_KEYWORDS = {
    "stealer", "hack", "trojan", "backdoor", "keylogger", "phish", "ransom", "malicious", "evil"
}

TEXT_SUSPICIOUS_TERMS = {
    "suspicious", "suspious", "suspected", "anomaly", "anomalous", "untrusted"
}

CRITICAL_TEXT_TERMS = {
    "stealer", "hack", "trojan", "backdoor", "keylogger", "phish", "malicious", "evil", "ransom"
}

NEGATION_TERMS = {"not", "no", "without", "isnt", "isn't", "na", "nai", "none"}

COMMON_TEXT_STOPWORDS = {
    "this", "that", "package", "and", "or", "is", "are", "was", "were", "looks", "look",
    "seems", "seem", "behavior", "activity", "issue", "normal", "safe", "verified", "official"
}

BENIGN_TEXT_HINTS = {
    "safe", "trusted", "official", "verified", "normal", "clean", "benign", "ok"
}

SAFE_ALTERNATIVES = {
    "requests": "httpx==0.27.0",
    "flask": "fastapi==0.115.0",
    "django": "fastapi==0.115.0",
    "numpy": "numpy==1.26.4",
    "pandas": "pandas==2.2.2",
}

CVE_DATABASE = {
    "requests": [
        {"id": "CVE-2023-32681", "severity": "MEDIUM", "description": "Proxy-Authorization header leak on redirect"},
        {"id": "CVE-2024-35195", "severity": "MEDIUM", "description": "Certificate verification bypass via incorrect netrc"},
    ],
    "pillow": [
        {"id": "CVE-2023-44271", "severity": "HIGH", "description": "Uncontrolled resource consumption via crafted images"},
    ],
    "cryptography": [
        {"id": "CVE-2023-49083", "severity": "LOW", "description": "NULL pointer dereference in PKCS12 parsing"},
    ],
}

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
DEFAULT_ML_FEATURES = [
    "syscall_count",
    "suspicious_connection_count",
    "file_access_count",
    "data_exfiltrated_kb",
    "process_spawn_count",
]

def load_selected_features() -> List[str]:
    """Load selected feature names from CSV header if present."""
    try:
        with open("final_selected_features.csv", "r", encoding="utf-8") as fp:
            header = fp.readline().strip()
        if not header:
            return DEFAULT_ML_FEATURES
        columns = [c.strip() for c in header.split(",") if c.strip()]
        return [c for c in columns if c.lower() != "label"] or DEFAULT_ML_FEATURES
    except Exception:
        return DEFAULT_ML_FEATURES

def get_model_feature_count() -> int:
    """Return expected feature count from model metadata."""
    if MODEL_LOADED and hasattr(model, "n_features_in_"):
        try:
            return int(getattr(model, "n_features_in_"))
        except Exception:
            pass
    return len(DEFAULT_ML_FEATURES)

def _string_to_number(value: str) -> float:
    """Convert a string token to a deterministic numeric value."""
    token = (value or "").strip()
    if token == "":
        return 0.0

    lowered = token.lower()
    direct_map = {
        "male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0,
        "true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0,
    }
    if lowered in direct_map:
        return direct_map[lowered]

    try:
        return float(token)
    except Exception:
        pass

    if len(token) == 1:
        return float(ord(token))

    hashed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
    return float(hashed % 1000)

def _coerce_value_to_float(value: Any) -> float:
    """Convert supported primitive values to float."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _string_to_number(value)
    return _string_to_number(str(value))

def _normalize_row_to_numeric(row: Any) -> List[float]:
    """Normalize one row (list or dict) into numeric feature list."""
    if isinstance(row, dict):
        values = list(row.values())
    elif isinstance(row, list):
        values = row
    else:
        raise HTTPException(400, "Each row must be a list or object")

    if not values:
        raise HTTPException(400, "Input row is empty")

    return [_coerce_value_to_float(v) for v in values]

def predict_from_feature_values(numeric_values: List[float]) -> dict:
    """Run prediction for a single numeric feature vector using trained model."""
    required = get_model_feature_count()
    original_count = len(numeric_values)

    if original_count < required:
        used_values = numeric_values + ([0.0] * (required - original_count))
    else:
        used_values = numeric_values[:required]
    dropped_values = numeric_values[required:]
    padded_missing_count = max(required - original_count, 0)

    if MODEL_LOADED and model is not None:
        try:
            # Scale features if scaler exists
            if SCALER_LOADED and scaler is not None:
                used_values_scaled = scaler.transform([used_values])[0]
            else:
                used_values_scaled = used_values

            pred = int(model.predict([used_values_scaled])[0])
            proba = None
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba([used_values_scaled])[0][1])

            score = int(round(proba * 100)) if proba is not None else (90 if pred == 1 else 10)
            label = "MALICIOUS" if pred == 1 else "BENIGN"
        except Exception as exc:
            raise HTTPException(500, f"Prediction failed: {exc}")
    else:
        # Fallback simulation when model not available
        score = max(0, min(100, int(sum(used_values) / max(len(used_values), 1))))
        label = "MALICIOUS" if score >= 60 else "SUSPICIOUS" if score >= 35 else "BENIGN"

    return {
        "label": label,
        "risk_score": score,
        "confidence": score,
        "received_features": original_count,
        "used_features": required,
        "padded_missing_count": padded_missing_count,
        "auto_padded_with_zeros": padded_missing_count > 0,
        "dropped_extra_values": dropped_values,
        "input_used": used_values,
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.utcnow().isoformat(),
    }

# ─── File Processing Helpers ─────────────────────────────────────────────────

def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX files."""
    if DOCX_AVAILABLE:
        try:
            doc = Document(BytesIO(content))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception:
            pass
    
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            xml_content = zf.read("word/document.xml").decode("utf-8", errors="ignore")
            text = re.sub(r"<[^>]+>", " ", xml_content)
            text = re.sub(r"\s+", " ", text).strip()
            return text
    except Exception:
        return ""

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF files."""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception:
        return ""

def extract_packages_from_excel(content: bytes, file_ext: str) -> List[str]:
    """Extract package names from Excel files."""
    packages = []
    
    if not OPENPYXL_AVAILABLE and not PANDAS_AVAILABLE:
        return packages
    
    try:
        if file_ext == ".xlsx" and OPENPYXL_AVAILABLE:
            wb = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
            sheet = wb.active
            
            for row in sheet.iter_rows(max_row=200, max_col=10, values_only=True):
                for cell in row:
                    if cell and isinstance(cell, str):
                        pkg_candidate = re.split(r"[>=<!~@\s]", cell.strip(), maxsplit=1)[0]
                        if pkg_candidate and len(pkg_candidate) > 2 and re.search(r'[a-zA-Z]', pkg_candidate):
                            packages.append(pkg_candidate.lower())
            wb.close()
        
        elif file_ext == ".xls" and PANDAS_AVAILABLE:
            df = pd.read_excel(BytesIO(content), engine='xlrd', nrows=200)
            for col in df.columns:
                for value in df[col].dropna().head(100):
                    if isinstance(value, str):
                        pkg_candidate = re.split(r"[>=<!~@\s]", value.strip(), maxsplit=1)[0]
                        if pkg_candidate and len(pkg_candidate) > 2 and re.search(r'[a-zA-Z]', pkg_candidate):
                            packages.append(pkg_candidate.lower())
    
    except Exception as e:
        print(f"Excel extraction error: {e}")
    
    return list(dict.fromkeys(packages))[:100]

def extract_packages_from_csv_text(content: str) -> List[str]:
    """Extract package names from CSV rows."""
    packages: List[str] = []
    header_tokens = {"package", "packages", "name", "dependency", "module", "library", "dependencies"}
    
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(content[:1024])
        reader = csv.DictReader(content.splitlines(), dialect=dialect)
        fieldnames = [f.lower().strip() for f in (reader.fieldnames or [])]
        
        package_columns = ["package", "name", "dependency", "module", "library", "dependencies"]
        selected_column = next((c for c in package_columns if c in fieldnames), None)
        
        for row in reader:
            if not row:
                continue
            
            value = ""
            if selected_column:
                for k, v in row.items():
                    if (k or "").lower().strip() == selected_column:
                        value = (v or "").strip()
                        break
            else:
                first_value = next(iter(row.values()), "")
                value = (first_value or "").strip()
            
            if not value:
                continue
            
            pkg = re.split(r"[>=<!~@\s]", value, maxsplit=1)[0].strip().lower()
            if pkg and len(pkg) > 1 and re.search(r'[a-zA-Z]', pkg):
                packages.append(pkg)
    
    except Exception as e:
        for line in content.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue

            first_cell = cleaned.split(',', 1)[0].strip()
            pkg = re.split(r"[>=<!~@\s]", first_cell, maxsplit=1)[0].strip().lower()

            if pkg in header_tokens:
                continue

            if pkg and len(pkg) > 1 and re.search(r'[a-zA-Z]', pkg):
                packages.append(pkg)
    
    return list(dict.fromkeys(packages))

def extract_packages_from_json(content: bytes) -> List[str]:
    """Extract package names from JSON files."""
    packages = []
    try:
        data = json.loads(content.decode('utf-8'))
        dep_fields = ['dependencies', 'devDependencies', 'peerDependencies', 'packages']
        
        def extract_from_obj(obj, depth=0):
            if depth > 10:
                return
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in dep_fields and isinstance(value, dict):
                        for pkg in value.keys():
                            if pkg and len(pkg) > 1:
                                packages.append(pkg.lower())
                    elif isinstance(value, (dict, list)):
                        extract_from_obj(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_from_obj(item, depth + 1)
        
        extract_from_obj(data)
    except Exception:
        pass
    
    return list(dict.fromkeys(packages))

def extract_packages_from_yaml(content: bytes) -> List[str]:
    """Extract package names from YAML files."""
    packages = []
    if not YAML_AVAILABLE:
        return packages
    
    try:
        data = yaml.safe_load(content.decode('utf-8'))
        
        def extract_from_obj(obj, depth=0):
            if depth > 10:
                return
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ['dependencies', 'packages', 'requires'] and isinstance(value, (dict, list)):
                        if isinstance(value, dict):
                            for pkg in value.keys():
                                if pkg and len(pkg) > 1:
                                    packages.append(pkg.lower())
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, str):
                                    pkg = re.split(r"[>=<!~@\s]", item, maxsplit=1)[0]
                                    if pkg and len(pkg) > 1:
                                        packages.append(pkg.lower())
                    elif isinstance(value, (dict, list)):
                        extract_from_obj(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_from_obj(item, depth + 1)
        
        extract_from_obj(data)
    except Exception:
        pass
    
    return list(dict.fromkeys(packages))

def extract_packages_from_tar(content: bytes) -> List[str]:
    """Extract package names from tar/tar.gz files."""
    packages = []
    manifest_files = ['requirements.txt', 'setup.py', 'pyproject.toml', 'package.json', 'Pipfile', 'poetry.lock']
    
    try:
        with tarfile.open(fileobj=BytesIO(content), mode='r:*') as tar:
            for member in tar.getmembers():
                basename = os.path.basename(member.name)
                if basename in manifest_files:
                    try:
                        f = tar.extractfile(member)
                        if f:
                            file_content = f.read().decode('utf-8', errors='ignore')
                            extracted = extract_packages_from_text(file_content)
                            packages.extend(extracted)
                    except Exception:
                        continue
    except Exception as e:
        print(f"Tar extraction error: {e}")
    
    return list(dict.fromkeys(packages))[:100]

def extract_packages_from_zip(content: bytes) -> List[str]:
    """Extract package names from zip files."""
    packages = []
    manifest_files = ['requirements.txt', 'setup.py', 'pyproject.toml', 'package.json', 'Pipfile', 'poetry.lock']
    
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            for name in zf.namelist():
                basename = os.path.basename(name)
                if basename in manifest_files:
                    try:
                        with zf.open(name) as f:
                            file_content = f.read().decode('utf-8', errors='ignore')
                            extracted = extract_packages_from_text(file_content)
                            packages.extend(extracted)
                    except Exception:
                        continue
    except Exception as e:
        print(f"Zip extraction error: {e}")
    
    return list(dict.fromkeys(packages))[:100]

def extract_packages_from_text(text: str) -> List[str]:
    """Extract package names from text content."""
    packages = []
    
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if '==' in line or '>=' in line or '<=' in line or '~=' in line:
            pkg = re.split(r'[>=<!~]', line, maxsplit=1)[0].strip()
        elif ',' in line:
            for part in line.split(','):
                pkg = part.strip().split()[0] if part.strip() else ''
                if pkg:
                    packages.append(pkg)
        else:
            pkg = line.split()[0] if line.split() else line
            packages.append(pkg)
    
    filtered = []
    for pkg in packages:
        pkg = pkg.strip().lower()
        if pkg and len(pkg) > 1 and len(pkg) < 100 and re.search(r'[a-zA-Z]', pkg):
            pkg = re.split(r'[>=<!~@]', pkg, maxsplit=1)[0]
            filtered.append(pkg)
    
    return list(dict.fromkeys(filtered))

def extract_package_from_filename(filename: str) -> str:
    """Extract package name from filename."""
    name = os.path.splitext(filename)[0]
    
    if filename.endswith('.whl'):
        parts = name.split('-')
        if parts:
            return parts[0].lower()
    
    if filename.endswith('.tar.gz'):
        name = name.replace('.tar', '')
    
    name = re.split(r'[-_][0-9]', name, maxsplit=1)[0]
    name = re.sub(r'[-_](v?\d+(?:\.\d+)*).*$', '', name)
    
    return name.lower()

# ─── Sandbox Simulation ──────────────────────────────────────────────────────

def simulate_sandbox_analysis(pkg_name: str, ecosystem: str) -> dict:
    """Simulates Docker sandbox behavioral analysis."""
    time.sleep(0.3)
    
    seed_material = f"{pkg_name.strip().lower()}|{ecosystem.strip().lower()}"
    seed_value = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed_value)
    
    normalized_name = pkg_name.lower()
    is_known_bad = normalized_name in KNOWN_MALICIOUS
    has_bad_keyword = any(kw in normalized_name for kw in SUSPICIOUS_KEYWORDS)
    has_critical_keyword = any(kw in normalized_name for kw in CRITICAL_PACKAGE_KEYWORDS)
    
    benign_syscalls = [
        "read(3, ...) = 4096",
        "write(1, ...) = 128",
        "mmap(NULL, 8192) = 0x7f...",
        "open('/etc/ld.so.cache', O_RDONLY) = 3",
        "stat('/usr/lib/python3', ...) = 0",
        "brk(NULL) = 0x...",
        "close(3) = 0",
        "fstat(1, ...) = 0",
    ]
    
    suspicious_syscalls = [
        "open('/etc/passwd', O_RDONLY) = 3",
        "open('/etc/shadow', O_RDONLY) = 4",
        "connect(4, {sa_family=AF_INET, sin_port=htons(4444)}) = 0",
        "execve('/bin/bash', ['bash', '-i']) = 0",
        "ptrace(PTRACE_ATTACH, 1, ...) = 0",
        "recvfrom(3, ..., MSG_PEEK) = 1024",
    ]
    
    if is_known_bad or has_critical_keyword:
        syscalls = rng.sample(suspicious_syscalls, k=min(4, len(suspicious_syscalls))) + \
                   rng.sample(benign_syscalls, k=3)
        network_connections = [
            {"ip": f"185.{rng.randint(1,255)}.{rng.randint(1,255)}.{rng.randint(1,255)}", "port": 4444, "suspicious": True},
            {"ip": f"103.{rng.randint(1,255)}.{rng.randint(1,255)}.{rng.randint(1,255)}", "port": 8080, "suspicious": True},
        ]
        data_exfiltrated_kb = rng.randint(50, 500)
        files_accessed = ["/etc/passwd", "/etc/shadow"]
        processes_spawned = ["bash -i", "sh -c curl"]
    elif has_bad_keyword:
        syscalls = rng.sample(suspicious_syscalls, k=1) + rng.sample(benign_syscalls, k=4)
        network_connections = [
            {"ip": f"172.{rng.randint(16,31)}.{rng.randint(1,255)}.{rng.randint(1,255)}", "port": 8080, "suspicious": True}
        ] if rng.randint(0, 1) == 1 else []
        data_exfiltrated_kb = rng.randint(0, 40)
        files_accessed = ["/usr/lib/python3/"]
        processes_spawned = []
    else:
        syscalls = rng.sample(benign_syscalls, k=rng.randint(3, 6))
        network_connections = []
        data_exfiltrated_kb = 0
        files_accessed = ["/usr/lib/python3/"]
        processes_spawned = []
    
    return {
        "syscall_count": len(syscalls) * rng.randint(10, 50),
        "syscall_samples": syscalls,
        "network_connections": network_connections,
        "data_exfiltrated_kb": data_exfiltrated_kb,
        "files_accessed": files_accessed,
        "processes_spawned": processes_spawned,
        "sandbox_duration_ms": rng.randint(800, 2500),
    }

def calculate_risk_score(sandbox_data: dict, pkg_name: str) -> dict:
    """Calculate ML-based risk score."""
    
    syscall_component = min(int(sandbox_data["syscall_count"] / 25), 14)
    duration_component = min(int(sandbox_data["sandbox_duration_ms"] / 500), 6)
    process_component = min(len(sandbox_data["processes_spawned"]) * 5, 12)
    score = syscall_component + duration_component + process_component
    factors = []
    
    cred_files = [f for f in sandbox_data["files_accessed"] if "/etc/passwd" in f or "/etc/shadow" in f]
    if cred_files:
        score += 35
        factors.append({"factor": "Credential file access (/etc/passwd, /etc/shadow)", "impact": 35, "type": "critical"})
    
    suspicious_nets = [n for n in sandbox_data["network_connections"] if n["suspicious"]]
    if suspicious_nets:
        score += 30
        factors.append({"factor": f"External C2 connections ({len(suspicious_nets)} suspicious IPs)", "impact": 30, "type": "critical"})
    
    if sandbox_data["data_exfiltrated_kb"] > 0:
        score += 20
        factors.append({"factor": f"Data exfiltration detected ({sandbox_data['data_exfiltrated_kb']} KB sent)", "impact": 20, "type": "high"})
    
    if sandbox_data["processes_spawned"]:
        score += 15
        factors.append({"factor": "Spawned shell processes (bash/sh)", "impact": 15, "type": "high"})
    
    if any(kw in pkg_name.lower() for kw in SUSPICIOUS_KEYWORDS):
        score += 20
        factors.append({"factor": "Package name contains malicious keyword", "impact": 20, "type": "medium"})
    
    if sandbox_data["syscall_count"] > 500:
        score += 10
        factors.append({"factor": f"Abnormally high syscall count ({sandbox_data['syscall_count']})", "impact": 10, "type": "medium"})
    
    if MODEL_LOADED and model is not None:
        try:
            feature_vector = [[
                sandbox_data["syscall_count"],
                len(suspicious_nets),
                len(sandbox_data["files_accessed"]),
                sandbox_data["data_exfiltrated_kb"],
                len(sandbox_data["processes_spawned"]),
            ]]
            
            if hasattr(model, "predict_proba"):
                ml_proba = float(model.predict_proba(feature_vector)[0][1])
                ml_score = int(round(ml_proba * 100))
                score = int(round((score * 0.75) + (ml_score * 0.25)))
            
            ml_pred = int(model.predict(feature_vector)[0])
            if ml_pred == 1:
                proba_hint = ml_proba if ml_proba is not None else 0.6
                ml_boost = min(max(int(round(proba_hint * 12)), 4), 12)
                score += ml_boost
                factors.append({"factor": "ML model flagged package as malicious", "impact": ml_boost, "type": "high"})
        except Exception:
            pass
    
    indicator_count = 0
    if cred_files:
        indicator_count += 1
    if suspicious_nets:
        indicator_count += 1
    if sandbox_data["data_exfiltrated_kb"] > 0:
        indicator_count += 1
    if sandbox_data["processes_spawned"]:
        indicator_count += 1
    if any(kw in pkg_name.lower() for kw in SUSPICIOUS_KEYWORDS):
        indicator_count += 1
    
    if indicator_count >= 2:
        suspicious_floor = 35 + min(5, len(suspicious_nets) + int(sandbox_data["data_exfiltrated_kb"] / 120))
        score = max(score, suspicious_floor)
    if indicator_count >= 3:
        score = max(score, 60)
    
    if indicator_count == 3:
        score = min(score, 89)
    elif indicator_count == 4:
        score = min(score, 95)
    
    if 35 <= score < 60:
        suspicious_delta = (
            min(int(sandbox_data["syscall_count"] / 180), 6)
            + min(len(suspicious_nets) * 2, 6)
            + min(int(sandbox_data["data_exfiltrated_kb"] / 90), 5)
            + min(len(sandbox_data["processes_spawned"]), 3)
        )
        score = min(59, max(35, score + suspicious_delta - 3))
    
    if indicator_count <= 2 and score >= 60:
        score = 59
    
    if indicator_count == 0:
        score = min(score, 34)
    
    score = min(score, 99)
    confidence_span = max(2, int((100 - score) / 20))
    confidence_low = max(score - confidence_span, 0)
    confidence_high = min(score + confidence_span, 100)
    
    label = "MALICIOUS" if score >= 60 else ("SUSPICIOUS" if score >= 35 else "BENIGN")
    
    return {
        "score": score,
        "label": label,
        "confidence_band": {"low": confidence_low, "high": confidence_high},
        "shap_factors": sorted(factors, key=lambda x: -x["impact"]),
    }

def get_ai_remediation(pkg_name: str, risk_data: dict, sandbox_data: dict) -> dict:
    """Generate AI remediation."""
    
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    
    if api_key:
        try:
            prompt = f"""You are a cybersecurity expert. Analyze this malicious package report and provide remediation.

Package: {pkg_name}
Risk Score: {risk_data['score']}% {risk_data['label']}
Suspicious Factors: {json.dumps(risk_data['shap_factors'], indent=2)}
Sandbox Findings:
- Syscalls: {sandbox_data['syscall_count']}
- Network connections: {sandbox_data['network_connections']}
- Files accessed: {sandbox_data['files_accessed']}
- Data exfiltrated: {sandbox_data['data_exfiltrated_kb']} KB

Respond in this exact JSON format:
{{
  "summary": "Brief 2-sentence explanation of what this package does maliciously",
  "threat_type": "One of: Data Stealer, Backdoor, Cryptominer, Typosquat, Supply Chain Attack",
  "safe_alternative": "package_name==version or null",
  "fix_steps": ["step1", "step2", "step3"],
  "cve_references": ["CVE-XXXX-YYYY or null"]
}}"""

            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {"sk-0d7cd4848f324f0cb611185f14edbc76"}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
                timeout=15
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                content = re.sub(r"```json|```", "", content).strip()
                return json.loads(content)
        except Exception as e:
            print(f"[DeepSeek Error] {e}")
    
    threat_type = "Backdoor" if any("connect" in s for s in sandbox_data["syscall_samples"]) else \
                  "Data Stealer" if sandbox_data["data_exfiltrated_kb"] > 0 else "Typosquat"
    
    alt = SAFE_ALTERNATIVES.get(pkg_name.lower(), None)
    cves = CVE_DATABASE.get(pkg_name.lower(), [])
    
    return {
        "summary": f"Package '{pkg_name}' exhibits malicious behavior including unauthorized system access and data exfiltration.",
        "threat_type": threat_type,
        "safe_alternative": alt,
        "fix_steps": [
            f"Immediately remove '{pkg_name}' from your environment",
            f"pip uninstall {pkg_name} -y",
            "Rotate any credentials that may have been exposed",
            "Check your network logs for outbound connections to suspicious IPs",
        ],
        "cve_references": [c["id"] for c in cves] if cves else [],
    }

def find_scan_by_id(scan_id: str) -> Optional[dict]:
    for scan in scan_history:
        if scan.get("scan_id") == scan_id:
            return scan
    return None

def build_scan_result(scan_id: str, pkg_name: str, ecosystem: str, sandbox: dict, risk: dict, extra: Optional[dict] = None) -> dict:
    result = {
        "scan_id": scan_id,
        "package_name": pkg_name,
        "ecosystem": ecosystem,
        "timestamp": datetime.utcnow().isoformat(),
        "sandbox": sandbox,
        "risk": risk,
        "remediation": get_ai_remediation(pkg_name, risk, sandbox) if risk["score"] >= 35 else None,
    }
    if extra:
        result.update(extra)
    return result

def analyze_package_for_batch(pkg: str, ecosystem: str) -> dict:
    """Analyze a single package for batch mode."""
    sandbox = simulate_sandbox_analysis(pkg, ecosystem)
    risk = calculate_risk_score(sandbox, pkg)
    return {
        "package": pkg,
        "risk_score": risk["score"],
        "label": risk["label"],
        "top_factor": risk["shap_factors"][0]["factor"] if risk["shap_factors"] else "None",
        "needs_action": risk["score"] >= 35,
    }

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "SCALA-Guard API",
        "version": "2.0.0",
        "status": "operational",
        "model_loaded": MODEL_LOADED,
        "model_accuracy": model_metrics.get('ensemble_accuracy', 0) if model_metrics else 0,
        "endpoints": [
            "/api/features",
            "/api/predict",
            "/api/predict/csv",
            "/api/model/stats",
            "/analyze",
            "/analyze/name",
            "/analyze/text",
            "/analyze/batch",
            "/history",
            "/stats"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "scaler_loaded": SCALER_LOADED,
        "features_loaded": FEATURES_LOADED,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/model/stats")
def get_model_stats():
    """Get model training statistics and performance metrics"""
    if model_metrics:
        return {
            "model_loaded": MODEL_LOADED,
            "performance": {
                "accuracy": model_metrics.get('ensemble_accuracy', 0),
                "f1_score": model_metrics.get('ensemble_f1_score', 0),
                "precision": model_metrics.get('ensemble_precision', 0),
                "recall": model_metrics.get('ensemble_recall', 0),
                "roc_auc": model_metrics.get('ensemble_auc', 0)
            },
            "class_distribution": model_metrics.get('class_distribution', {}),
            "features_used": len(feature_cols),
            "best_individual_model": model_metrics.get('best_individual_model', 'Unknown'),
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "model_loaded": MODEL_LOADED,
            "error": "Model metrics not available",
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/features")
def get_features():
    """Return feature metadata for frontend clients."""
    return {
        "count": len(feature_cols),
        "features": feature_cols,
        "model_loaded": MODEL_LOADED,
    }

@app.post("/api/predict")
def predict_numeric(payload: Union[List[float], dict] = Body(...)):
    """Predict from numeric features."""
    if isinstance(payload, dict):
        data = payload.get("data")
        if data is None:
            data = payload.get("features")
        if data is None:
            data = payload.get("values")
    else:
        data = payload

    if data is None:
        raise HTTPException(400, "Missing input data")

    # Batch mode
    if isinstance(data, list) and data and isinstance(data[0], (list, dict)):
        predictions = []
        errors = []
        for idx, row in enumerate(data, start=1):
            try:
                numeric_values = _normalize_row_to_numeric(row)
                result = predict_from_feature_values(numeric_values)
                predictions.append({"row": idx, **result})
            except HTTPException as exc:
                errors.append({"row": idx, "error": str(exc.detail)})

        if not predictions:
            raise HTTPException(400, "No valid rows to predict")

        return {
            "mode": "batch",
            "total_predicted": len(predictions),
            "total_errors": len(errors),
            "errors": errors,
            "predictions": predictions,
        }

    numeric_values = _normalize_row_to_numeric(data)
    return predict_from_feature_values(numeric_values)

@app.post("/api/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """Upload a local CSV file and run row-wise numeric predictions."""
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded CSV is empty")

    if len(raw) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(413, "CSV file too large. Max supported size is 20MB")

    try:
        text = raw.decode("utf-8-sig", errors="ignore")
    except Exception:
        raise HTTPException(400, "Failed to decode CSV file")

    reader = csv.reader(io.StringIO(text))
    predictions = []
    skipped_rows = []

    for row_idx, row in enumerate(reader, start=1):
        values = [cell.strip() for cell in row if str(cell).strip() != ""]
        if not values:
            continue

        try:
            numeric_values = [_coerce_value_to_float(v) for v in values]
        except Exception:
            if row_idx == 1:
                continue
            skipped_rows.append({"row": row_idx, "reason": "non-numeric values"})
            continue

        try:
            pred = predict_from_feature_values(numeric_values)
            predictions.append({"row": row_idx, **pred})
        except HTTPException as exc:
            skipped_rows.append({"row": row_idx, "reason": str(exc.detail)})

    if not predictions:
        raise HTTPException(400, "No valid numeric rows found in CSV")

    return {
        "filename": filename,
        "total_predicted": len(predictions),
        "total_skipped": len(skipped_rows),
        "skipped_rows": skipped_rows[:100],
        "predictions": predictions,
    }

@app.post("/analyze")
async def analyze_file(package_file: UploadFile = File(...)):
    """Analyze an uploaded package file."""
    allowed_exts = {".whl", ".tar.gz", ".tgz", ".zip", ".gz", ".pdf", ".csv", 
                    ".doc", ".docx", ".xlsx", ".xls", ".json", ".yaml", ".yml",
                    ".txt", ".requirements", ".in", ".lock"}
    
    filename = os.path.basename(package_file.filename or "").strip()
    if not filename:
        raise HTTPException(400, "Filename is required")
    
    file_ext = None
    for ext in allowed_exts:
        if filename.endswith(ext):
            file_ext = ext
            break
    
    if not file_ext:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(sorted(allowed_exts))}")
    
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{filename}"
    content = await package_file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(413, "File too large. Max supported size is 20MB")
    with open(path, "wb") as f:
        f.write(content)
    
    pkg_name = extract_package_from_filename(filename)
    extracted_packages = []
    
    try:
        if file_ext in [".docx", ".doc"]:
            text = extract_text_from_docx(content)
            if text:
                extracted_packages = extract_packages_from_text(text)
        elif file_ext == ".pdf":
            text = extract_text_from_pdf(content)
            if text:
                extracted_packages = extract_packages_from_text(text)
        elif file_ext in [".xlsx", ".xls"]:
            extracted_packages = extract_packages_from_excel(content, file_ext)
        elif file_ext == ".csv":
            try:
                csv_text = content.decode("utf-8")
                extracted_packages = extract_packages_from_csv_text(csv_text)
            except Exception:
                pass
        elif file_ext == ".json":
            extracted_packages = extract_packages_from_json(content)
        elif file_ext in [".yaml", ".yml"]:
            extracted_packages = extract_packages_from_yaml(content)
        elif file_ext in [".tar.gz", ".tgz", ".gz"]:
            extracted_packages = extract_packages_from_tar(content)
        elif file_ext == ".zip":
            extracted_packages = extract_packages_from_zip(content)
        elif file_ext in [".txt", ".requirements", ".in", ".lock"]:
            try:
                text_content = content.decode("utf-8")
                extracted_packages = extract_packages_from_text(text_content)
            except Exception:
                pass
    except Exception as e:
        print(f"Extraction error: {e}")
    
    if extracted_packages:
        pkg_name = extracted_packages[0]
    
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    
    sandbox = simulate_sandbox_analysis(pkg_name, "pypi")
    risk = calculate_risk_score(sandbox, pkg_name)
    
    result = build_scan_result(
        scan_id=f"sg-{int(time.time())}-{file_hash[:8]}",
        pkg_name=pkg_name,
        ecosystem="pypi",
        sandbox=sandbox,
        risk=risk,
        extra={
            "filename": filename,
            "file_type": file_ext,
            "file_size_kb": round(len(content) / 1024, 2),
            "file_hash_sha256": file_hash,
            "extracted_packages_count": len(extracted_packages),
            "extracted_packages": extracted_packages[:20] if extracted_packages else []
        }
    )
    
    scan_history.append(result)
    
    try:
        os.remove(path)
    except Exception:
        pass
    
    return result

@app.post("/analyze/name")
async def analyze_by_name(req: PackageScanRequest):
    """Analyze a package by name."""
    pkg_name = req.name.strip().lower()
    if not pkg_name:
        raise HTTPException(400, "Package name is required")
    
    registry_info = {}
    try:
        if req.ecosystem == "pypi":
            url = f"https://pypi.org/pypi/{pkg_name}/json"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                info = data.get("info", {})
                registry_info = {
                    "name": info.get("name", pkg_name),
                    "version": info.get("version", "unknown"),
                    "author": info.get("author", "Unknown"),
                    "summary": info.get("summary", ""),
                    "home_page": info.get("home_page", ""),
                }
        elif req.ecosystem == "npm":
            url = f"https://registry.npmjs.org/{pkg_name}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                latest_version = data.get("dist-tags", {}).get("latest", "unknown")
                registry_info = {
                    "name": data.get("name", pkg_name),
                    "version": latest_version,
                    "author": str(data.get("author", "Unknown")),
                    "summary": data.get("description", ""),
                    "home_page": data.get("homepage", ""),
                }
    except Exception as e:
        registry_info = {"error": str(e)}
    
    sandbox = simulate_sandbox_analysis(pkg_name, req.ecosystem)
    risk = calculate_risk_score(sandbox, pkg_name)
    
    result = build_scan_result(
        scan_id=f"sg-{int(time.time())}-{hashlib.md5(pkg_name.encode()).hexdigest()[:8]}",
        pkg_name=pkg_name,
        ecosystem=req.ecosystem,
        sandbox=sandbox,
        risk=risk,
        extra={
            "version": req.version,
            "registry_info": registry_info,
        }
    )
    
    scan_history.append(result)
    return result

@app.post("/analyze/text")
async def analyze_text(req: TextScanRequest):
    """Analyze free text."""
    payload = req.text.strip()
    if not payload:
        raise HTTPException(400, "Text is required")
    
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{1,64}", payload.lower())
    matched = [t for t in tokens if t in KNOWN_MALICIOUS or any(k in t for k in SUSPICIOUS_KEYWORDS)]
    
    negated_critical_terms = set()
    for idx, token in enumerate(tokens):
        if token not in CRITICAL_TEXT_TERMS:
            continue
        left_window = tokens[max(0, idx - 2):idx]
        if any(w in NEGATION_TERMS for w in left_window):
            negated_critical_terms.add(token)
    
    matched_suspicious_hits = []
    for t in tokens:
        if not any(k in t for k in SUSPICIOUS_KEYWORDS):
            continue
        if t in negated_critical_terms:
            continue
        matched_suspicious_hits.append(t)
    
    text_suspicion_hits = [t for t in tokens if t in TEXT_SUSPICIOUS_TERMS]
    text_signal_hits = list(dict.fromkeys(text_suspicion_hits + matched_suspicious_hits))
    has_critical_text = any((term in payload.lower()) and (term not in negated_critical_terms) for term in CRITICAL_TEXT_TERMS)
    has_known_malicious_token = any(t in KNOWN_MALICIOUS for t in tokens)
    benign_hits = [t for t in tokens if t in BENIGN_TEXT_HINTS]
    
    matched_effective = [t for t in matched if t not in negated_critical_terms]
    if matched_effective:
        pkg_name = matched_effective[0]
    else:
        pkg_candidates = [
            t for t in tokens
            if t not in COMMON_TEXT_STOPWORDS
            and t not in NEGATION_TERMS
            and t not in negated_critical_terms
            and len(t) > 2
        ]
        pkg_name = pkg_candidates[0] if pkg_candidates else "unknown-package"
    
    sandbox = simulate_sandbox_analysis(pkg_name, req.ecosystem)
    risk = calculate_risk_score(sandbox, pkg_name)
    
    if text_signal_hits and not benign_hits and risk["score"] < 35:
        adjusted = 37 + min(len(text_signal_hits) * 3, 12)
        risk["score"] = min(adjusted, 59)
        risk["label"] = "SUSPICIOUS"
        risk.setdefault("shap_factors", []).append({
            "factor": "Suspicious indicators detected in text input",
            "impact": 15,
            "type": "medium",
        })
        span = max(2, int((100 - risk["score"]) / 20))
        risk["confidence_band"] = {
            "low": max(risk["score"] - span, 0),
            "high": min(risk["score"] + span, 100),
        }
    
    if text_signal_hits and not benign_hits and not has_critical_text and not has_known_malicious_token and risk["score"] >= 60:
        adjusted = 45 + min(len(text_signal_hits) * 4, 12)
        risk["score"] = min(adjusted, 59)
        risk["label"] = "SUSPICIOUS"
        span = max(2, int((100 - risk["score"]) / 20))
        risk["confidence_band"] = {
            "low": max(risk["score"] - span, 0),
            "high": min(risk["score"] + span, 100),
        }
    
    result = build_scan_result(
        scan_id=f"txt-{int(time.time())}-{hashlib.md5(payload.encode()).hexdigest()[:8]}",
        pkg_name=pkg_name,
        ecosystem=req.ecosystem,
        sandbox=sandbox,
        risk=risk,
        extra={
            "source": "text",
            "text_length": len(payload),
            "matched_indicators": list(dict.fromkeys(matched))[:20],
            "text_suspicion_hits": list(dict.fromkeys(text_suspicion_hits))[:20],
            "matched_suspicious_hits": list(dict.fromkeys(matched_suspicious_hits))[:20],
            "benign_text_hits": list(dict.fromkeys(benign_hits))[:20],
            "negated_critical_terms": list(dict.fromkeys(list(negated_critical_terms)))[:20],
        }
    )
    
    scan_history.append(result)
    return result

@app.post("/analyze/batch")
async def analyze_batch_file(package_file: UploadFile = File(...), ecosystem: str = "pypi"):
    """Scan entire requirements.txt, package.json, CSV, Excel, or other dependency files."""
    
    filename = (package_file.filename or "").lower()
    raw_content = await package_file.read()
    if not raw_content:
        raise HTTPException(400, "Uploaded file is empty")
    
    content = raw_content.decode("utf-8", errors="ignore")
    packages = []
    
    if filename.endswith("package.json"):
        packages = extract_packages_from_json(raw_content)
    elif filename.endswith(".csv"):
        packages = extract_packages_from_csv_text(content)
    elif filename.endswith((".xlsx", ".xls")):
        packages = extract_packages_from_excel(raw_content, f".{filename.split('.')[-1]}")
    elif filename.endswith((".yaml", ".yml")):
        packages = extract_packages_from_yaml(raw_content)
    else:
        packages = extract_packages_from_text(content)
    
    packages = list(dict.fromkeys([p for p in packages if p and len(p) > 1]))
    
    if not packages:
        raise HTTPException(400, "No packages found in the uploaded file")

    tasks = [asyncio.to_thread(analyze_package_for_batch, pkg, ecosystem) for pkg in packages]
    results = await asyncio.gather(*tasks)
    
    summary = {
        "total": len(results),
        "malicious": len([r for r in results if r["label"] == "MALICIOUS"]),
        "suspicious": len([r for r in results if r["label"] == "SUSPICIOUS"]),
        "benign": len([r for r in results if r["label"] == "BENIGN"]),
    }
    
    batch_result = {
        "scan_id": f"batch-{int(time.time())}",
        "filename": package_file.filename,
        "ecosystem": ecosystem,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary,
        "results": sorted(results, key=lambda x: -x["risk_score"]),
    }
    
    scan_history.append(batch_result)
    return batch_result

@app.get("/history")
def get_history(limit: int = 20):
    return {
        "total": len(scan_history),
        "scans": list(reversed(scan_history))[:limit]
    }

@app.get("/history/{scan_id}")
def get_scan(scan_id: str):
    scan = find_scan_by_id(scan_id)
    if not scan:
        raise HTTPException(404, "Scan not found")
    return scan

@app.delete("/history/{scan_id}")
def delete_scan(scan_id: str):
    for idx, scan in enumerate(scan_history):
        if scan.get("scan_id") == scan_id:
            removed = scan_history.pop(idx)
            return {"message": "Scan deleted", "scan_id": removed.get("scan_id")}
    raise HTTPException(404, "Scan not found")

@app.get("/stats")
def get_stats():
    total = len(scan_history)
    malicious = sum(1 for s in scan_history if s.get("risk", {}).get("label") == "MALICIOUS")
    suspicious = sum(1 for s in scan_history if s.get("risk", {}).get("label") == "SUSPICIOUS")
    
    return {
        "total_scans": total,
        "malicious_found": malicious,
        "suspicious_found": suspicious,
        "benign": total - malicious - suspicious,
        "threat_rate": round((malicious / total * 100) if total > 0 else 0, 1),
    }

@app.delete("/history")
def clear_history():
    scan_history.clear()
    return {"message": "History cleared"}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("SCALA-Guard API Server")
    print("="*70)
    print(f"\n✅ Model Loaded: {MODEL_LOADED}")
    print(f"✅ Scaler Loaded: {SCALER_LOADED}")
    print(f"✅ Features Loaded: {FEATURES_LOADED}")
    if model_metrics:
        print(f"📊 Model Accuracy: {model_metrics.get('ensemble_accuracy', 0):.2%}")
    print(f"\n🚀 Server starting at: http://localhost:8000")
    print(f"📖 API Documentation: http://localhost:8000/docs")
    print(f"🔧 Features API: http://localhost:8000/api/features")
    print(f"📊 Model Stats: http://localhost:8000/api/model/stats")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)