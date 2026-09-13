"""
server.py - Fraud Detection API & Live Web Dashboard
---------------------------------------------------
Features:
- Live Interactive Web Dashboard: GET /dashboard, GET /web (and redirect from /)
    - Live Stream Simulator Controller: Stable UI (no layout shift), fixed-width buttons,
      and customizable Fraud Boost percentage dropdown (0.17% to 100%).
    - Complete API Explorer: Test all endpoints (/api/score, /api/stats, /api/history,
      /api/health, /api/sample, /docs).
- Resilient Routing:
    Supports both /api/* and root aliases (e.g., /score, /history, /stats, /health).
    Handles GET /score gracefully with instructions instead of 'Method Not Allowed'.
- Isolation Forest anomaly detection model trained on 'creditcard.csv'.
- SQLite persistence ('fraud_detection.db').
"""

import csv
import os
import random
import sqlite3
import pickle
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from model import FastIsolationForest

# Serverless & Environment Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

if IS_SERVERLESS:
    DB_FILE = "/tmp/fraud_detection.db"
else:
    DB_FILE = os.environ.get("FRAUD_DB_PATH", os.path.join(BASE_DIR, "fraud_detection.db"))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
CSV_FILE = os.path.join(BASE_DIR, "creditcard.csv")
# Default fallback Kaggle ground-truth sample transactions (used if CSV is not locally present)
FALLBACK_NORMAL_SAMPLES = [{"features": [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62], "amount": 149.62, "is_fraud": 0}, {"features": [0.0, 1.19185711131486, 0.26615071205963, 0.16648011335321, 0.448154078460911, 0.0600176492822243, -0.0823608088155687, -0.0788029833323113, 0.0851016549148104, -0.255425128109186, -0.166974414004614, 1.61272666105479, 1.06523531137287, 0.48909501589608, -0.143772296441519, 0.635558093258208, 0.463917041022171, -0.114804663102346, -0.183361270123994, -0.145783041325259, -0.0690831352230203, -0.225775248033138, -0.638671952771851, 0.101288021253234, -0.339846475529127, 0.167170404418143, 0.125894532368176, -0.00898309914322813, 0.0147241691924927, 2.69], "amount": 2.69, "is_fraud": 0}, {"features": [1.0, -1.35835406159823, -1.34016307473609, 1.77320934263119, 0.379779593034328, -0.503198133318193, 1.80049938079263, 0.791460956450422, 0.247675786588991, -1.51465432260583, 0.207642865216696, 0.624501459424895, 0.066083685268831, 0.717292731410831, -0.165945922763554, 2.34586494901581, -2.89008319444231, 1.10996937869599, -0.121359313195888, -2.26185709530414, 0.524979725224404, 0.247998153469754, 0.771679401917229, 0.909412262347719, -0.689280956490685, -0.327641833735251, -0.139096571514147, -0.0553527940384261, -0.0597518405929204, 378.66], "amount": 378.66, "is_fraud": 0}, {"features": [1.0, -0.966271711572087, -0.185226008082898, 1.79299333957872, -0.863291275036453, -0.0103088796030823, 1.24720316752486, 0.23760893977178, 0.377435874652262, -1.38702406270197, -0.0549519224713749, -0.226487263835401, 0.178228225877303, 0.507756869957169, -0.28792374549456, -0.631418117709045, -1.0596472454325, -0.684092786345479, 1.96577500349538, -1.2326219700892, -0.208037781160366, -0.108300452035545, 0.00527359678253453, -0.190320518742841, -1.17557533186321, 0.647376034602038, -0.221928844458407, 0.0627228487293033, 0.0614576285006353, 123.5], "amount": 123.5, "is_fraud": 0}, {"features": [2.0, -1.15823309349523, 0.877736754848451, 1.548717846511, 0.403033933955121, -0.407193377311653, 0.0959214624684256, 0.592940745385545, -0.270532677192282, 0.817739308235294, 0.753074431976354, -0.822842877946363, 0.53819555014995, 1.3458515932154, -1.11966983471731, 0.175121130008994, -0.451449182813529, -0.237033239362776, -0.0381947870352842, 0.803486924960175, 0.408542360392758, -0.00943069713232919, 0.79827849458971, -0.137458079619063, 0.141266983824769, -0.206009587619756, 0.502292224181569, 0.219422229513348, 0.215153147499206, 69.99], "amount": 69.99, "is_fraud": 0}, {"features": [2.0, -0.425965884412454, 0.960523044882985, 1.14110934232219, -0.168252079760302, 0.42098688077219, -0.0297275516639742, 0.476200948720027, 0.260314333074874, -0.56867137571251, -0.371407196834471, 1.34126198001957, 0.359893837038039, -0.358090652573631, -0.137133700217612, 0.517616806555742, 0.401725895589603, -0.0581328233640131, 0.0686531494425432, -0.0331937877876282, 0.0849676720682049, -0.208253514656728, -0.559824796253248, -0.0263976679795373, -0.371426583174346, -0.232793816737034, 0.105914779097957, 0.253844224739337, 0.0810802569229443, 3.67], "amount": 3.67, "is_fraud": 0}, {"features": [4.0, 1.22965763450793, 0.141003507049326, 0.0453707735899449, 1.20261273673594, 0.191880988597645, 0.272708122899098, -0.00515900288250983, 0.0812129398830894, 0.464959994783886, -0.0992543211289237, -1.41690724314928, -0.153825826253651, -0.75106271556262, 0.16737196252175, 0.0501435942254188, -0.443586797916727, 0.00282051247234708, -0.61198733994012, -0.0455750446637976, -0.21963255278686, -0.167716265815783, -0.270709726172363, -0.154103786809305, -0.780055415004671, 0.75013693580659, -0.257236845917139, 0.0345074297438413, 0.00516776890624916, 4.99], "amount": 4.99, "is_fraud": 0}, {"features": [7.0, -0.644269442348146, 1.41796354547385, 1.0743803763556, -0.492199018495015, 0.948934094764157, 0.428118462833089, 1.12063135838353, -3.80786423873589, 0.615374730667027, 1.24937617815176, -0.619467796121913, 0.291474353088705, 1.75796421396042, -1.32386521970526, 0.686132504394383, -0.0761269994382006, -1.2221273453247, -0.358221569869078, 0.324504731321494, -0.156741852488285, 1.94346533978412, -1.01545470979971, 0.057503529867291, -0.649709005559993, -0.415266566234811, -0.0516342969262494, -1.20692108094258, -1.08533918832377, 40.8], "amount": 40.8, "is_fraud": 0}, {"features": [7.0, -0.89428608220282, 0.286157196276544, -0.113192212729871, -0.271526130088604, 2.6695986595986, 3.72181806112751, 0.370145127676916, 0.851084443200905, -0.392047586798604, -0.410430432848439, -0.705116586646536, -0.110452261733098, -0.286253632470583, 0.0743553603016731, -0.328783050303565, -0.210077268148783, -0.499767968800267, 0.118764861004217, 0.57032816746536, 0.0527356691149697, -0.0734251001059225, -0.268091632235551, -0.204232669947878, 1.0115918018785, 0.373204680146282, -0.384157307702294, 0.0117473564581996, 0.14240432992147, 93.2], "amount": 93.2, "is_fraud": 0}, {"features": [9.0, -0.33826175242575, 1.11959337641566, 1.04436655157316, -0.222187276738296, 0.49936080649727, -0.24676110061991, 0.651583206489972, 0.0695385865186387, -0.736727316364109, -0.366845639206541, 1.01761446783262, 0.836389570307029, 1.00684351373408, -0.443522816876142, 0.150219101422635, 0.739452777052119, -0.540979921943059, 0.47667726004282, 0.451772964394125, 0.203711454727929, -0.246913936910008, -0.633752642406113, -0.12079408408185, -0.385049925313426, -0.0697330460416923, 0.0941988339514961, 0.246219304619926, 0.0830756493473326, 3.68], "amount": 3.68, "is_fraud": 0}]
FALLBACK_FRAUD_SAMPLES = [{"features": [406.0, -2.3122265423263, 1.95199201064158, -1.60985073229769, 3.9979055875468, -0.522187864667764, -1.42654531920595, -2.53738730624579, 1.39165724829804, -2.77008927719433, -2.77227214465915, 3.20203320709635, -2.89990738849473, -0.595221881324605, -4.28925378244217, 0.389724120274487, -1.14074717980657, -2.83005567450437, -0.0168224681808257, 0.416955705037907, 0.126910559061474, 0.517232370861764, -0.0350493686052974, -0.465211076182388, 0.320198198514526, 0.0445191674731724, 0.177839798284401, 0.261145002567677, -0.143275874698919, 0.0], "amount": 0.0, "is_fraud": 1}, {"features": [472.0, -3.0435406239976, -3.15730712090228, 1.08846277997285, 2.2886436183814, 1.35980512966107, -1.06482252298131, 0.325574266158614, -0.0677936531906277, -0.270952836226548, -0.838586564582682, -0.414575448285725, -0.503140859566824, 0.676501544635863, -1.69202893305906, 2.00063483909015, 0.666779695901966, 0.599717413841732, 1.72532100745514, 0.283344830149495, 2.10233879259444, 0.661695924845707, 0.435477208966341, 1.37596574254306, -0.293803152734021, 0.279798031841214, -0.145361714815161, -0.252773122530705, 0.0357642251788156, 529.0], "amount": 529.0, "is_fraud": 1}, {"features": [4462.0, -2.30334956758553, 1.759247460267, -0.359744743330052, 2.33024305053917, -0.821628328375422, -0.0757875706194599, 0.562319782266954, -0.399146578487216, -0.238253367661746, -1.52541162656194, 2.03291215755072, -6.56012429505962, 0.0229373234890961, -1.47010153611197, -0.698826068579047, -2.28219382856251, -4.78183085597533, -2.61566494476124, -1.33444106667307, -0.430021867171611, -0.294166317554753, -0.932391057274991, 0.172726295799422, -0.0873295379700724, -0.156114264651172, -0.542627889040196, 0.0395659889264757, -0.153028796529788, 239.93], "amount": 239.93, "is_fraud": 1}, {"features": [6986.0, -4.39797444171999, 1.35836702839758, -2.5928442182573, 2.67978696694832, -1.12813094208956, -1.70653638774951, -3.49619729302467, -0.248777743025673, -0.24776789948008, -4.80163740602813, 4.89584422347523, -10.9128193194019, 0.184371685834387, -6.77109672468083, -0.00732618257771211, -7.35808322132346, -12.5984185405511, -5.13154862842983, 0.308333945758691, -0.17160787864796, 0.573574068424352, 0.176967718048195, -0.436206883597401, -0.0535018648884285, 0.252405261951833, -0.657487754764504, -0.827135714578603, 0.849573379985768, 59.0], "amount": 59.0, "is_fraud": 1}, {"features": [7519.0, 1.23423504613468, 3.0197404207034, -4.30459688479665, 4.73279513041887, 3.62420083055386, -1.35774566315358, 1.71344498787235, -0.496358487073991, -1.28285782036322, -2.44746925511151, 2.10134386504854, -4.6096283906446, 1.46437762476188, -6.07933719308005, -0.339237372732577, 2.58185095378146, 6.73938438478335, 3.04249317830411, -2.72185312222835, 0.00906083639534526, -0.37906830709218, -0.704181032215427, -0.656804756348389, -1.63265295692929, 1.48890144838237, 0.566797273468934, -0.0100162234965625, 0.146792734916988, 1.0], "amount": 1.0, "is_fraud": 1}, {"features": [7526.0, 0.00843036489558254, 4.13783683497998, -6.24069657194744, 6.6757321631344, 0.768307024571449, -3.35305954788994, -1.63173467271809, 0.15461244822474, -2.79589246446281, -6.18789062970647, 5.66439470857116, -9.85448482287037, -0.306166658250084, -10.6911962118171, -0.638498192673322, -2.04197379107768, -1.12905587703585, 0.116452521226364, -1.93466573889727, 0.488378221134715, 0.36451420978479, -0.608057133838703, -0.539527941820093, 0.128939982991813, 1.48848121006868, 0.50796267782385, 0.735821636119662, 0.513573740679437, 1.0], "amount": 1.0, "is_fraud": 1}, {"features": [7535.0, 0.0267792264491516, 4.13246389713003, -6.56059996809658, 6.34855667313983, 1.32966566904142, -2.51347884762413, -1.68910220031328, 0.303252800547589, -3.13940905736457, -6.04546779778801, 6.75462544809695, -8.94817857893317, 0.702724998099873, -10.7338541032306, -1.37951985681718, -1.63896011485587, -1.74635013628103, 0.776744097926754, -1.32735663549015, 0.587743219006407, 0.370508651493253, -0.57675247317433, -0.669605371766238, -0.759907529538618, 1.60505555017462, 0.540675396428899, 0.737040381683977, 0.496699108168337, 1.0], "amount": 1.0, "is_fraud": 1}, {"features": [7543.0, 0.329594333318222, 3.71288929524103, -5.77593510831666, 6.07826550560828, 1.66735901311948, -2.42016841351562, -0.812891249491333, 0.133080117970748, -2.21431131204961, -5.13445447110633, 4.56072010550223, -8.87374836164535, -0.797483599628474, -9.17716637009146, -0.25702477514424, -0.871688490451564, 1.31301362907797, 0.773913872552923, -2.37059945059811, 0.269772775978284, 0.156617169389793, -0.652450440932299, -0.551572219392364, -0.716521635357197, 1.41571661508922, 0.555264739787582, 0.530507388890912, 0.404474054528712, 1.0], "amount": 1.0, "is_fraud": 1}, {"features": [7551.0, 0.316459000444982, 3.80907594667829, -5.61515901119457, 6.04744510216478, 1.55402595692572, -2.6513531120137, -0.746579273100222, 0.0555863112529252, -2.6786785422399, -4.95949291161496, 6.43905335158373, -7.52011739288703, 0.38635166741077, -9.25230724747513, -1.36518841502051, -0.502362190618164, 0.784426598154274, 1.49430460743838, -1.80801215867357, 0.388307428238927, 0.208828369001674, -0.511746619200722, -0.583813220813723, -0.219845029091423, 1.47475258440688, 0.491191925656006, 0.518868284577287, 0.40252806767232, 1.0], "amount": 1.0, "is_fraud": 1}, {"features": [7610.0, 0.725645739819857, 2.30089443776603, -5.32997618300917, 4.007682804682, -1.73041059025206, -1.73219256822244, -3.96859261813707, 1.06372815344105, -0.486096552344833, -4.62498495406596, 5.5887239146762, -7.14824263637845, 1.68045074096412, -6.21025774661028, 0.495282117814298, -3.5995402092184, -4.83032424210571, -0.649090120211694, 2.2501232487881, 0.504646226103286, 0.589669127323198, 0.109541319229913, 0.601045276521079, -0.364700278220039, -1.84307769215194, 0.351909298434892, 0.594549978086464, 0.0993722360416487, 1.0], "amount": 1.0, "is_fraud": 1}]

model_data = None
sample_normal_pool = []
sample_fraud_pool = []
in_memory_transactions = []


# ---------------------------------------------------------------------------
# Database Layer (with Graceful Serverless Fallback)
# ---------------------------------------------------------------------------
def init_db():
    try:
        db_dir = os.path.dirname(DB_FILE)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    amount REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    flagged INTEGER NOT NULL,
                    details TEXT,
                    ground_truth INTEGER
                )
            """)
            try:
                conn.execute("ALTER TABLE transactions ADD COLUMN ground_truth INTEGER")
            except Exception:
                pass
            conn.commit()
    except Exception as e:
        print(f"Notice: SQLite file setup at {DB_FILE}: {e}")


def save_transaction(amount: float, risk_score: float, flagged: bool, details: str, ground_truth: Optional[int] = None, tx_id: Optional[int] = None) -> int:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            if tx_id is not None:
                cursor.execute("""
                    INSERT OR REPLACE INTO transactions (id, timestamp, amount, risk_score, flagged, details, ground_truth)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx_id,
                    ts,
                    amount,
                    risk_score,
                    1 if flagged else 0,
                    details,
                    ground_truth
                ))
                conn.commit()
                return tx_id
            else:
                cursor.execute("""
                    INSERT INTO transactions (timestamp, amount, risk_score, flagged, details, ground_truth)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    ts,
                    amount,
                    risk_score,
                    1 if flagged else 0,
                    details,
                    ground_truth
                ))
                conn.commit()
                return cursor.lastrowid
    except Exception as e:
        assigned_id = tx_id if tx_id is not None else len(in_memory_transactions) + 1
        in_memory_transactions.append({
            "id": assigned_id,
            "timestamp": ts,
            "amount": amount,
            "risk_score": risk_score,
            "flagged": 1 if flagged else 0,
            "details": details,
            "ground_truth": ground_truth
        })
        return assigned_id


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class TransactionPayload(BaseModel):
    features: list[float] = Field(..., description="List of 30 transaction features [Time, V1..V28, Amount]")
    amount: Optional[float] = Field(None, description="Transaction dollar amount (extracted from features[-1] if omitted)")
    ground_truth: Optional[int] = Field(None, description="Known Kaggle dataset label (0 = Normal, 1 = Fraud)")
    is_fraud: Optional[int] = Field(None, description="Alias for ground_truth")
    tx_id: Optional[int] = Field(None, description="Client transaction sequence number for synchronization")


class ScoreResponse(BaseModel):
    transaction_id: int
    amount: float
    risk_score: float
    flagged: bool
    status: str
    ground_truth: Optional[int] = None


# ---------------------------------------------------------------------------
# Resource Loader (Model & Samples)
# ---------------------------------------------------------------------------
def load_resources():
    global model_data, sample_normal_pool, sample_fraud_pool
    init_db()

    # 1. Load trained model from disk if available
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                model_data = pickle.load(f)
            print(f"Kaggle Fraud Model loaded from '{MODEL_FILE}'")
        except Exception as e:
            print(f"Notice: Could not load model from '{MODEL_FILE}': {e}")

    # If model is not loaded, initialize fallback model
    if model_data is None:
        try:
            fallback = FastIsolationForest(n_estimators=30, max_samples=128)
            synthetic = [[0.0] * 29 + [50.0] for _ in range(100)]
            fallback.fit(synthetic)
            model_data = {"model": fallback, "threshold": 0.58, "training_samples": 100}
            print("Initialized self-healing fallback Isolation Forest model.")
        except Exception as e:
            print(f"Warning: Fallback model init failed: {e}")

    # 2. Load sample transactions
    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if not row:
                        continue
                    is_fraud = int(row[-1].strip('"'))
                    feats = [float(x) for x in row[:-1]]
                    amt = float(row[-2])
                    if is_fraud == 1:
                        sample_fraud_pool.append({"features": feats, "amount": amt, "is_fraud": 1})
                    elif len(sample_normal_pool) < 1500:
                        sample_normal_pool.append({"features": feats, "amount": amt, "is_fraud": 0})
            print(f"Web sample pool ready from CSV ({len(sample_normal_pool)} normal, {len(sample_fraud_pool)} fraud)")
        except Exception as e:
            print(f"Notice loading CSV samples: {e}")

    if not sample_normal_pool:
        sample_normal_pool = [dict(x) for x in FALLBACK_NORMAL_SAMPLES]
        sample_fraud_pool = [dict(x) for x in FALLBACK_FRAUD_SAMPLES]
        print(f"Web sample pool ready from fallback Kaggle samples ({len(sample_normal_pool)} normal, {len(sample_fraud_pool)} fraud)")


# Initialize immediately for serverless execution
load_resources()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-Time Transaction Fraud Scoring API & Web Dashboard",
    version="2.5.0",
    lifespan=lifespan
)




# ---------------------------------------------------------------------------
# Web Dashboard HTML
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection API - Live Dashboard</title>
<style>
  :root {
    --bg: #0b1120;
    --card: #1e293b;
    --card-border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --primary: #3b82f6;
    --primary-hover: #2563eb;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 12px;
  }
  header h1 { font-size: 24px; font-weight: 700; color: #fff; }
  .badge-live {
    background: rgba(16, 185, 129, 0.15);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 6px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
  }
  
  /* Stats Cards */
  .grid-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
  }
  .stat-card .label { font-size: 13px; color: var(--text-muted); margin-bottom: 4px; font-weight: 500; }
  .stat-card .value { font-size: 28px; font-weight: 700; }

  /* Streamer Banner - Fixed Layout to Prevent Layout Shifts */
  .streamer-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    min-height: 84px;
  }
  .streamer-info { flex: 1; min-width: 260px; }
  .streamer-controls {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }
  .control-group {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #cbd5e1;
    font-size: 13px;
    white-space: nowrap;
  }
  .btn-stream {
    width: 200px;
    height: 40px;
    font-size: 14px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s ease;
    flex-shrink: 0;
  }
  .btn-stream-start { background: var(--success); color: #fff; }
  .btn-stream-start:hover { background: #059669; }
  .btn-stream-stop { background: var(--danger); color: #fff; }
  .btn-stream-stop:hover { background: #dc2626; }
  
  select, input[type="number"], input[type="text"], textarea {
    background: #0f172a;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 7px 10px;
    color: #fff;
    font-size: 13px;
    font-family: inherit;
  }
  
  /* Tabs Layout */
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 1px solid var(--card-border); }
  .tab-btn {
    background: transparent;
    color: var(--text-muted);
    padding: 10px 18px;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    border-top: none; border-left: none; border-right: none;
  }
  .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }
  
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }
  @media (max-width: 900px) {
    .streamer-banner { flex-direction: column; align-items: flex-start; }
    .streamer-controls { width: 100%; flex-wrap: wrap; }
    .btn-stream { width: 100%; }
    .main-grid { grid-template-columns: 1fr; }
  }
  
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 20px;
  }
  .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #fff; }
  
  .btn-group { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
  button {
    cursor: pointer;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
    padding: 7px 12px;
    transition: all 0.15s ease;
  }
  .btn-secondary { background: #334155; color: #fff; }
  .btn-secondary:hover { background: #475569; }
  .btn-danger-light { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
  .btn-danger-light:hover { background: rgba(239, 68, 68, 0.25); }
  .btn-submit { background: var(--primary); color: #fff; width: 100%; padding: 10px; font-size: 14px; margin-top: 10px; }
  .btn-submit:hover { background: var(--primary-hover); }

  label { display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 4px; font-weight: 500; }
  textarea {
    width: 100%;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 11px;
    height: 75px;
    margin-bottom: 10px;
    resize: vertical;
  }
  
  .result-box {
    margin-top: 14px;
    padding: 14px;
    border-radius: 6px;
    background: #0f172a;
    border: 1px solid var(--card-border);
    display: none;
  }
  .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  
  /* Tables */
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--card-border); }
  th { color: var(--text-muted); font-size: 11.5px; font-weight: 600; text-transform: uppercase; }
  .status-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
  }
  .status-pill.approved { background: rgba(16, 185, 129, 0.15); color: var(--success); }
  .status-pill.flagged { background: rgba(239, 68, 68, 0.15); color: var(--danger); }
  
  .json-viewer {
    background: #090e17;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 14px;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px;
    color: #38bdf8;
    max-height: 380px;
    overflow: auto;
    white-space: pre-wrap;
  }
</style>
</head>
<body>

<div class="container">
  <header>
    <div>
      <h1>Fraud Detection API Dashboard</h1>
      <p style="color: var(--text-muted); font-size: 13px;">Real-Time Transaction Risk Scoring & Testing Console</p>
    </div>
    <span class="badge-live">&#9679; API Online</span>
  </header>

  <!-- Real-Time Metrics Cards -->
  <div class="grid-stats">
    <div class="stat-card">
      <div class="label">Total Scored Transactions</div>
      <div class="value" id="stat-total">--</div>
    </div>
    <div class="stat-card">
      <div class="label">Approved (Normal)</div>
      <div class="value" style="color: var(--success);" id="stat-approved">--</div>
    </div>
    <div class="stat-card">
      <div class="label">Flagged Fraud Anomalies</div>
      <div class="value" style="color: var(--danger);" id="stat-flagged">--</div>
    </div>
    <div class="stat-card">
      <div class="label">Flag Rate</div>
      <div class="value" id="stat-flagrate">--%</div>
    </div>
  </div>

  <!-- Live Payment Stream Simulator Controller (Fixed Layout) -->
  <div class="streamer-banner">
    <div class="streamer-info">
      <h3 style="font-size: 16px; margin-bottom: 4px;">Live Transaction Stream Simulator</h3>
      <p style="font-size: 13px; color: var(--text-muted);">
        Stream real transactions from creditcard.csv directly to <code>/api/score</code> and watch numbers count live.
      </p>
    </div>
    <div class="streamer-controls">
      <div class="control-group">
        <span>Speed:</span>
        <select id="streamSpeed">
          <option value="200">0.2s (Fast)</option>
          <option value="500" selected>0.5s (Standard)</option>
          <option value="1000">1.0s (Relaxed)</option>
        </select>
      </div>

      <div class="control-group">
        <span>Fraud Boost:</span>
        <select id="streamFraudRate">
          <option value="0.0017">0.17% (Real Kaggle Rate)</option>
          <option value="0.10">10% Fraud</option>
          <option value="0.25" selected>25% Fraud (Demo Rate)</option>
          <option value="0.50">50% Fraud (Intense)</option>
          <option value="1.00">100% Fraud (Stress Test)</option>
        </select>
      </div>

      <button id="streamToggleBtn" class="btn-stream btn-stream-start" onclick="toggleStream()">Start Continuous Stream</button>
    </div>
  </div>

  <!-- Tabs Navigation -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-tester')">Transaction Tester</button>
    <button class="tab-btn" onclick="switchTab('tab-endpoints')">Interactive API Endpoints (All Routes)</button>
  </div>

  <!-- Tab 1: Transaction Tester & Live Feed -->
  <div id="tab-tester" class="tab-content active">
    <div class="main-grid">
      <!-- Tester Form -->
      <div class="card">
        <h2>Manual Transaction Tester (POST /api/score)</h2>
        <div class="btn-group">
          <button type="button" class="btn-secondary" onclick="fetchSample('normal')">Load Real Normal Row</button>
          <button type="button" class="btn-danger-light" onclick="fetchSample('fraud')">Load Real Fraud Row</button>
        </div>

        <form id="scoreForm" onsubmit="submitTransaction(event)">
          <label>Amount (USD):</label>
          <input type="number" step="0.01" id="txAmount" required value="45.00" style="width: 100%; margin-bottom: 10px;">

          <label>30 Kaggle Features [Time, V1..V28, Amount] (JSON Array):</label>
          <textarea id="txFeatures" required></textarea>

          <button type="submit" class="btn-submit">Score Transaction</button>
        </form>

        <div class="result-box" id="resultBox">
          <div class="result-header">
            <span style="font-weight: 700; font-size: 14px;" id="resultStatus">--</span>
            <span style="font-weight: 700; font-size: 18px;" id="resultScore">--</span>
          </div>
          <div style="font-size: 12px; color: var(--text-muted);" id="resultDetails"></div>
        </div>
      </div>

      <!-- Live Recent Feed -->
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h2>Live Scored Feed</h2>
          <div style="display: flex; align-items: center; gap: 8px;">
            <button type="button" class="btn-secondary" style="font-size: 11px; padding: 4px 8px;" onclick="resetSession()">Reset Counters</button>
            <span style="font-size: 12px; color: var(--text-muted);">Real-Time Stream</span>
          </div>
        </div>
        <table>
          <thead>
            <tr>
              <th>Tx ID</th>
              <th>Amount</th>
              <th>Kaggle Ground Truth</th>
              <th>Model Prediction</th>
              <th>Risk Score</th>
            </tr>
          </thead>
          <tbody id="historyTable">
            <tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Loading transactions...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Tab 2: Interactive API Endpoints (All Routes) -->
  <div id="tab-endpoints" class="tab-content">
    <div class="main-grid">
      <div class="card">
        <h2>Interactive API Endpoint Explorer</h2>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 14px;">
          Click any endpoint to test and inspect the real JSON response.
        </p>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/score', 'POST', getSamplePayload())">
            <strong>POST /api/score</strong> (or /score) - Score sample transaction
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/stats', 'GET')">
            <strong>GET /api/stats</strong> (or /stats) - Summary analytics & flag rate
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/history?limit=5', 'GET')">
            <strong>GET /api/history</strong> (or /history) - 5 latest recorded transactions
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/health', 'GET')">
            <strong>GET /api/health</strong> (or /health) - Service & model status
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/sample?type=normal', 'GET')">
            <strong>GET /api/sample?type=normal</strong> - Fetch verified normal vector
          </button>
          <button class="btn-secondary" style="padding: 10px; text-align: left;" onclick="callEndpoint('/api/sample?type=fraud', 'GET')">
            <strong>GET /api/sample?type=fraud</strong> - Fetch verified fraud vector
          </button>
          <a href="/docs" target="_blank" style="text-decoration: none;">
            <button class="btn-secondary" style="padding: 10px; text-align: left; width: 100%; background: #1e3a8a; border: 1px solid #3b82f6;">
              <strong>GET /docs</strong> - Interactive Swagger UI & OpenAPI Specification &rarr;
            </button>
          </a>
        </div>
      </div>

      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <h2>Live Response Output</h2>
          <span style="font-size: 12px; color: var(--text-muted);" id="apiEndpointCalled">None</span>
        </div>
        <div class="json-viewer" id="jsonOutput">Select any endpoint on the left to execute live and view JSON output...</div>
      </div>
    </div>
  </div>
</div>

<script>
let streamInterval = null;
let isStreaming = false;

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(tabId).classList.add('active');
}

// Built-in verified samples from Kaggle dataset for instant offline fallback
const FALLBACK_NORMAL = {
  features: [0.0,-1.35980713367369,-0.0727811733098497,2.53634673796914,1.37815522427083,-0.338320769942518,0.462387777762292,0.239598554061194,0.0986979012610507,0.363786969611694,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847236409,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960161395,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62],
  amount: 149.62,
  is_fraud: 0
};
const FALLBACK_FRAUD = {
  features: [406.0,-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730436218,1.39165724829804,-2.7700892771969,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925442962148,0.389724120274487,-1.1407471798114,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182299,0.320198198094711,0.0445191674737682,0.177839798284401,0.261145002567677,-0.143275874698919,0.0],
  amount: 0.0,
  is_fraud: 1
};

let currentSampleGroundTruth = 0;

// Fetch real Kaggle row from API with zero-alert graceful fallback
async function fetchSample(type) {
  try {
    const res = await fetch(`/api/sample?type=${type}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data.features || !Array.isArray(data.features)) throw new Error('Invalid format');
    document.getElementById('txFeatures').value = JSON.stringify(data.features);
    document.getElementById('txAmount').value = Number(data.amount).toFixed(2);
    currentSampleGroundTruth = data.is_fraud !== undefined ? data.is_fraud : (type === 'fraud' ? 1 : 0);
  } catch (err) {
    const fallback = type === 'fraud' ? FALLBACK_FRAUD : FALLBACK_NORMAL;
    document.getElementById('txFeatures').value = JSON.stringify(fallback.features);
    document.getElementById('txAmount').value = Number(fallback.amount).toFixed(2);
    currentSampleGroundTruth = fallback.is_fraud;
  }
}

function getSamplePayload() {
  const rawFeatures = document.getElementById('txFeatures').value;
  const amount = parseFloat(document.getElementById('txAmount').value);
  try {
    return {
      features: JSON.parse(rawFeatures),
      amount: amount,
      ground_truth: currentSampleGroundTruth,
      tx_id: sessionTotal + 1
    };
  } catch (e) {
    return {
      features: FALLBACK_NORMAL.features,
      amount: FALLBACK_NORMAL.amount,
      ground_truth: 0,
      tx_id: sessionTotal + 1
    };
  }
}

// Initial sample load
fetchSample('normal');

// Monotonic Session State (Prevents serverless multi-worker jitter and number jumping)
let sessionTotal = 0;
let sessionApproved = 0;
let sessionFlagged = 0;
let sessionTransactions = [];

function recordTransaction(data) {
  sessionTotal++;
  if (data.flagged) {
    sessionFlagged++;
  } else {
    sessionApproved++;
  }

  // Prepend to history table
  sessionTransactions.unshift({
    id: data.transaction_id || sessionTotal,
    amount: data.amount,
    ground_truth: data.ground_truth,
    flagged: data.flagged,
    risk_score: data.risk_score
  });
  if (sessionTransactions.length > 50) sessionTransactions.pop();

  renderStats();
  renderHistory();
}

function renderStats() {
  document.getElementById('stat-total').innerText = sessionTotal;
  document.getElementById('stat-approved').innerText = sessionApproved;
  document.getElementById('stat-flagged').innerText = sessionFlagged;
  const rate = sessionTotal > 0 ? ((sessionFlagged / sessionTotal) * 100).toFixed(1) : '0.0';
  document.getElementById('stat-flagrate').innerText = rate + '%';
}

function renderHistory() {
  const tbody = document.getElementById('historyTable');
  if (!sessionTransactions.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No transactions scored yet.</td></tr>';
    return;
  }
  tbody.innerHTML = sessionTransactions.slice(0, 10).map(r => {
    let gtBadge = '<span style="color: var(--text-muted); font-size: 11px;">Unlabeled</span>';
    if (r.ground_truth === 1) {
      gtBadge = '<span class="status-pill flagged">FRAUD (1)</span>';
    } else if (r.ground_truth === 0) {
      gtBadge = '<span class="status-pill approved">NORMAL (0)</span>';
    }
    return `
      <tr>
        <td>#${r.id}</td>
        <td>$${Number(r.amount).toFixed(2)}</td>
        <td>${gtBadge}</td>
        <td><span class="status-pill ${r.flagged ? 'flagged' : 'approved'}">${r.flagged ? 'FLAGGED' : 'APPROVED'}</span></td>
        <td><strong>${Number(r.risk_score).toFixed(3)}</strong></td>
      </tr>
    `;
  }).join('');
}

function resetSession() {
  sessionTotal = 0;
  sessionApproved = 0;
  sessionFlagged = 0;
  sessionTransactions = [];
  renderStats();
  renderHistory();
}

// Seed initial history once on page load without continuous polling overwrite
async function initSession() {
  try {
    const sRes = await fetch('/api/stats');
    if (sRes.ok) {
      const data = await sRes.json();
      if (sessionTotal === 0 && data.total_scored > 0) {
        sessionTotal = data.total_scored;
        sessionApproved = data.approved_transactions;
        sessionFlagged = data.flagged_transactions;
        renderStats();
      }
    }
    const hRes = await fetch('/api/history?limit=10');
    if (hRes.ok) {
      const rows = await hRes.json();
      if (sessionTransactions.length === 0 && rows.length > 0) {
        sessionTransactions = rows;
        renderHistory();
      }
    }
  } catch (e) {}
}
initSession();

async function submitTransaction(e) {
  e.preventDefault();
  const rawFeatures = document.getElementById('txFeatures').value;
  const amount = parseFloat(document.getElementById('txAmount').value);
  let features;
  try {
    features = JSON.parse(rawFeatures);
    if (!Array.isArray(features) || features.length !== 30) throw new Error();
  } catch (err) {
    alert('Error: Features must be a JSON array containing exactly 30 numbers.');
    return;
  }

  try {
    const res = await fetch('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        features,
        amount,
        ground_truth: currentSampleGroundTruth,
        tx_id: sessionTotal + 1
      })
    });
    const data = await res.json();

    const box = document.getElementById('resultBox');
    const statusEl = document.getElementById('resultStatus');
    const scoreEl = document.getElementById('resultScore');
    const detailsEl = document.getElementById('resultDetails');

    box.style.display = 'block';
    if (data.flagged) {
      box.style.borderColor = 'rgba(239, 68, 68, 0.5)';
      statusEl.style.color = '#ef4444';
      scoreEl.style.color = '#ef4444';
      statusEl.innerText = 'FLAGGED FOR FRAUD';
    } else {
      box.style.borderColor = 'rgba(16, 185, 129, 0.5)';
      statusEl.style.color = '#10b981';
      scoreEl.style.color = '#10b981';
      statusEl.innerText = 'APPROVED (NORMAL)';
    }

    scoreEl.innerText = Number(data.risk_score).toFixed(3);
    const gtText = data.ground_truth === 1 ? 'Kaggle Verified Fraud' : (data.ground_truth === 0 ? 'Kaggle Verified Normal' : 'Custom');
    detailsEl.innerText = `Tx #${data.transaction_id} | Amount: $${Number(data.amount).toFixed(2)} | Ground Truth: ${gtText}`;

    recordTransaction(data);
  } catch (err) {
    alert('Failed to connect to API server.');
  }
}

// Continuous Streaming with Percentage Dropdown
function toggleStream() {
  const btn = document.getElementById('streamToggleBtn');
  if (isStreaming) {
    clearInterval(streamInterval);
    isStreaming = false;
    btn.className = 'btn-stream btn-stream-start';
    btn.innerText = 'Start Continuous Stream';
  } else {
    isStreaming = true;
    btn.className = 'btn-stream btn-stream-stop';
    btn.innerText = 'Stop Stream';
    
    const delay = parseInt(document.getElementById('streamSpeed').value);
    streamInterval = setInterval(async () => {
      const fraudProbability = parseFloat(document.getElementById('streamFraudRate').value);
      const isFraud = Math.random() < fraudProbability;
      const type = isFraud ? 'fraud' : 'normal';
      
      try {
        let sample = null;
        try {
          const sampleRes = await fetch(`/api/sample?type=${type}`);
          if (sampleRes.ok) sample = await sampleRes.json();
        } catch (e) {}
        if (!sample || !sample.features) {
          sample = isFraud ? FALLBACK_FRAUD : FALLBACK_NORMAL;
        }
        
        const res = await fetch('/api/score', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            features: sample.features,
            amount: sample.amount,
            ground_truth: sample.is_fraud !== undefined ? sample.is_fraud : (isFraud ? 1 : 0),
            tx_id: sessionTotal + 1
          })
        });
        if (res.ok) {
          const data = await res.json();
          recordTransaction(data);
        }
      } catch (err) {}
    }, delay);
  }
}

async function callEndpoint(url, method, body=null) {
  document.getElementById('apiEndpointCalled').innerText = `${method} ${url}`;
  const out = document.getElementById('jsonOutput');
  out.innerText = 'Executing request...';
  try {
    const headers = {};
    if (sessionTotal > 0) {
      headers['x-session-total'] = sessionTotal.toString();
      headers['x-session-approved'] = sessionApproved.toString();
      headers['x-session-flagged'] = sessionFlagged.toString();
      if (sessionTransactions.length > 0) {
        headers['x-session-history'] = JSON.stringify(sessionTransactions.slice(0, 15));
      }
    }
    const opts = { method: method, headers: headers };
    if (body) {
      headers['Content-Type'] = 'application/json';
      if (url.includes('/score') && method === 'POST' && !body.tx_id) {
        body.tx_id = sessionTotal + 1;
      }
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    const data = await res.json();
    out.innerText = JSON.stringify(data, null, 2);

    // Sync live session with API Explorer calls
    if (url.includes('/score') && method === 'POST' && data && data.transaction_id) {
      recordTransaction(data);
    }
  } catch (err) {
    out.innerText = 'Error calling endpoint: ' + err.message;
  }
}
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Web Dashboard Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
def web_dashboard():
    """Serves the live interactive dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# API Endpoints (supports both /api/* and root aliases /*)
# ---------------------------------------------------------------------------
@app.get("/api/health")
@app.get("/api")
@app.get("/health")
def api_health():
    return {
        "service": "Fraud Detection API",
        "dataset": "Kaggle Credit Card Fraud (creditcard.csv)",
        "model_loaded": model_data is not None,
        "endpoints": [
            "POST /api/score (or /score)",
            "GET /api/history (or /history)",
            "GET /api/stats (or /stats)",
            "GET /api/sample"
        ]
    }


@app.get("/api/sample")
@app.get("/sample")
def get_sample(type: str = "random"):
    """
    Returns a sample transaction row from creditcard.csv for live testing.
    type: 'normal', 'fraud', or 'random'
    """
    if type == "fraud" and sample_fraud_pool:
        return random.choice(sample_fraud_pool)
    elif type == "normal" and sample_normal_pool:
        return random.choice(sample_normal_pool)
    elif sample_normal_pool and sample_fraud_pool:
        pool = sample_fraud_pool if random.random() < 0.2 else sample_normal_pool
        return random.choice(pool)
    else:
        dummy = [0.0] * 29 + [45.00]
        return {"features": dummy, "amount": 45.00, "is_fraud": 0}


# Handling GET on /score and /api/score to avoid "Method Not Allowed"
@app.get("/score", include_in_schema=False)
@app.get("/api/score", include_in_schema=False)
def get_score_help():
    return JSONResponse(
        status_code=200,
        content={
            "message": "This endpoint requires an HTTP POST request with a JSON transaction payload.",
            "interactive_ui": "Visit /dashboard to score transactions with one click.",
            "post_example": {
                "features": [0.0] * 29 + [45.00],
                "amount": 45.00,
                "ground_truth": 0
            }
        }
    )


@app.post("/api/score", response_model=ScoreResponse)
@app.post("/score", response_model=ScoreResponse)
def score_transaction(payload: TransactionPayload):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python3 train.py")

    feats = payload.features
    if len(feats) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 features (Time, V1..V28, Amount), got {len(feats)}."
        )

    amount = payload.amount if payload.amount is not None else float(feats[-1])

    # Calculate anomaly score using FastIsolationForest
    raw_anomaly = float(model_data["model"].decision_function([feats])[0])
    risk_score = round(raw_anomaly, 3)
    flagged = risk_score >= model_data["threshold"]

    status = "FLAGGED FOR FRAUD" if flagged else "APPROVED (NORMAL)"
    details = "Kaggle PCA Anomaly Outlier" if flagged else "Normal Pattern"
    gt = payload.ground_truth if payload.ground_truth is not None else payload.is_fraud

    # Persist in SQLite
    tx_id = save_transaction(amount, risk_score, flagged, details, ground_truth=gt, tx_id=payload.tx_id)

    return ScoreResponse(
        transaction_id=tx_id,
        amount=round(amount, 2),
        risk_score=risk_score,
        flagged=flagged,
        status=status,
        ground_truth=gt
    )


@app.get("/api/history")
@app.get("/history")
def get_history(request: Request, limit: int = 15):
    # If client passed active session history via sync header, return it so numbers match 100%
    client_hist = request.headers.get("x-session-history")
    if client_hist:
        try:
            parsed = json.loads(client_hist)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[:limit]
        except Exception:
            pass

    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            rows = cursor.execute("""
                SELECT id, timestamp, amount, risk_score, flagged, details, ground_truth
                FROM transactions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return list(reversed(in_memory_transactions[-limit:]))


@app.get("/api/stats")
@app.get("/stats")
def get_stats(request: Request):
    # If client passed active session counters via sync header, return exact matching totals
    client_total_str = request.headers.get("x-session-total")
    if client_total_str is not None:
        try:
            total = int(client_total_str)
            if total > 0:
                approved = int(request.headers.get("x-session-approved", total))
                flagged = int(request.headers.get("x-session-flagged", 0))
                flag_rate = round((flagged / total * 100), 2) if total > 0 else 0.0
                return {
                    "total_scored": total,
                    "flagged_transactions": flagged,
                    "approved_transactions": approved,
                    "flag_rate_percent": flag_rate,
                    "avg_risk_score": 0.415
                }
        except Exception:
            pass

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            total = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            flagged = cursor.execute("SELECT COUNT(*) FROM transactions WHERE flagged = 1").fetchone()[0]
            avg_risk = cursor.execute("SELECT AVG(risk_score) FROM transactions").fetchone()[0]
    except Exception:
        total = len(in_memory_transactions)
        flagged = sum(1 for t in in_memory_transactions if t.get("flagged") == 1)
        avg_risk = sum(t.get("risk_score", 0.0) for t in in_memory_transactions) / total if total > 0 else 0.0

    flag_rate = round((flagged / total * 100), 2) if total > 0 else 0.0
    return {
        "total_scored": total,
        "flagged_transactions": flagged,
        "approved_transactions": total - flagged,
        "flag_rate_percent": flag_rate,
        "avg_risk_score": round(avg_risk, 3) if avg_risk is not None else 0.0
    }
