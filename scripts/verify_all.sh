#!/usr/bin/env bash
# dam-ai T14 verify_all — 零信任链尾复验（kanban t_2a55fc92）
#
# 不信任任何卡的 summary/metadata，换命令重跑五项核心断言：
#   C1 (T2)  openai SDK 纯文本 embeddings + cosine(self) >= 0.9999  (embedder :8090)
#   C2 (T5)  tagger 混合批失败隔离 + >64 输入拒绝                    (tagger   :8091)
#   C3 (T7)  任取 5 图 tools:1020 vs dam-ai:8092 逐字节对照          (ifa-test 容器内)
#   C4 (T13) 语义搜索 + 颜色搜索 + 打标链路在线可用                  (ifa-test 容器内)
#   C5 (T10) 泄露审计：shipped surface 无 soujpg/凭据/内网 IP
#
# 可重复执行；结果追加落 reports/verify-all.md（每次运行覆盖为本次结果）。
# 退出码 0 当且仅当全部 [check] PASS。
#
# 环境变量可覆盖端点：DAMAI_VERIFY_EMBEDDER / DAMAI_VERIFY_TAGGER /
# DAMAI_VERIFY_CLASSIC / DAMAI_VERIFY_TOOLS（默认按 PLAN.md 端口分配）。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="$REPO/reports/verify-all.md"
EMBEDDER="${DAMAI_VERIFY_EMBEDDER:-http://gpu0.dev.yufei.com:8090}"
TAGGER="${DAMAI_VERIFY_TAGGER:-http://gpu7.dev.yufei.com:8091}"
CLASSIC="${DAMAI_VERIFY_CLASSIC:-http://gpu7.dev.yufei.com:8092}"
TOOLS="${DAMAI_VERIFY_TOOLS:-http://gpu7.dev.yufei.com:1020}"
IFA="${DAMAI_VERIFY_IFA:-ifa-test}"
SJ_PY="${DAMAI_VERIFY_SJPY:-/data/apps/miniconda3/envs/sj/bin/python3}"

PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY=python3
PASS=0; FAIL=0
HEAD_SHA="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

say()  { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
emit() { say "$*"; echo "$*" >>"$REPORT"; }
# ck <id> <0|1> <detail>
ck() {
  local id="$1" ok="$2" detail="$3"
  if [ "$ok" = "1" ]; then emit "[check] PASS $id — $detail"; PASS=$((PASS+1));
  else emit "[check] FAIL $id — $detail"; FAIL=$((FAIL+1)); fi
}

: >"$REPORT"
{
  echo "# dam-ai verify_all — T14 零信任复验"
  echo "run: $(date '+%F %T')  repo HEAD: $HEAD_SHA  host: $(hostname)"
  echo "endpoints: embedder=$EMBEDDER tagger=$TAGGER classic=$CLASSIC tools=$TOOLS"
  echo
} >>"$REPORT"
say "verify_all start (HEAD=$HEAD_SHA)"

# ---- C0: wait for embedder model ready (24G clip-vit-l14 loads minutes) --------
C0_READY=0
for _ in $(seq 1 60); do
  if curl -sm 5 "$EMBEDDER/readyz" | grep -q '"status": *"ready"'; then C0_READY=1; break; fi
  sleep 10
done
ck "C0-embedder-ready-wait" "$C0_READY" "embedder /readyz status=ready within 600s [$EMBEDDER]"

# ---- C1: T2 openai SDK pure-text + cosine(self) -------------------------------
C1_OUT="$("$PY" - "$EMBEDDER" <<'PYEOF' 2>&1
import math, sys
from openai import OpenAI
c = OpenAI(base_url=sys.argv[1] + "/v1", api_key="empty")
r = c.embeddings.create(model="clip-vit-l14", input=["a red apple", "a red apple"])
v1, v2 = r.data[0].embedding, r.data[1].embedding
cos = sum(a*b for a, b in zip(v1, v2)) / (
    math.sqrt(sum(a*a for a in v1)) * math.sqrt(sum(a*a for a in v2)))
print(f"C1RESULT dims={len(v1)} obj={r.data[0].object} model={r.model} cos={cos:.6f}")
PYEOF
)"
C1_DIMS="$(sed -n 's/.*dims=\([0-9]*\).*/\1/p' <<<"$C1_OUT" | tail -1)"
C1_COS="$(sed -n 's/.*cos=\([0-9.]*\).*/\1/p' <<<"$C1_OUT" | tail -1)"
if awk -v c="$C1_COS" 'BEGIN{exit !(c>=0.9999)}' && [ "$C1_DIMS" = "768" ]; then C1_OK=1; else C1_OK=0; fi
ck "C1-t2-sdk-text-cosine" "$C1_OK" \
   "openai SDK text embeddings dims=$C1_DIMS cos(self)=$C1_COS (expect dims=768 cos>=0.9999) [$EMBEDDER]"

# ---- C2: T5 tagger isolation + 65 reject --------------------------------------
C2_OUT="$("$PY" - "$TAGGER" <<'PYEOF' 2>&1
import base64, json, sys, urllib.error, urllib.request
base = sys.argv[1]
png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
uri = "data:image/png;base64," + base64.b64encode(png).decode()
def post(body):
    req = urllib.request.Request(base + "/v1/tagging",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
st, body = post({"task": "image_caption_metadata", "inputs": [
    {"id": "good", "image_url": uri}, {"id": "bad", "image_url": "not-a-valid-url"}]})
res = body.get("results", [])
iso = (len(res) == 2 and res[0]["status"] == "ok" and res[1]["status"] == "error"
       and res[0]["id"] == "good" and res[1]["id"] == "bad")
st65, body65 = post({"task": "image_caption_metadata",
                     "inputs": [{"id": j, "image_url": uri} for j in range(65)]})
rej = st65 == 400 and body65.get("error", {}).get("code") == "batch_too_large"
print(f"C2RESULT iso={iso} rej={rej}")
PYEOF
)"
C2_ISO="$(sed -n 's/.*iso=\(True\|False\).*/\1/p' <<<"$C2_OUT" | tail -1)"
C2_REJ="$(sed -n 's/.*rej=\(True\|False\).*/\1/p' <<<"$C2_OUT" | tail -1)"
if [ "$C2_ISO" = "True" ] && [ "$C2_REJ" = "True" ]; then C2_OK=1; else C2_OK=0; fi
ck "C2-t5-isolation-cap65" "$C2_OK" \
   "tagger mixed-batch isolation=$C2_ISO, 65-input rejected as batch_too_large=$C2_REJ [$TAGGER]"

# ---- C3: T7 byte-diff 5 fresh images (tools vs dam-ai, in ifa-test) ------------
cp "$REPO/reports/t_2a55fc92-t7verify.py" "/tmp/t14_t7verify.py"
docker cp "/tmp/t14_t7verify.py" "$IFA:/tmp/t14_t7verify.py" >/dev/null || {
  docker cp "$REPO/reports/t_2a55fc92-t7verify.py" "$IFA:/tmp/t14_t7verify.py" >/dev/null; }
C3_OUT="$(timeout 560 docker exec "$IFA" "$SJ_PY" /tmp/t14_t7verify.py 2>&1 | grep -E '^T7VERIFY')"
C3_SUM="$(sed -n 's/.*T7VERIFY-SUMMARY pass=\([0-9]\)\/\([0-9]\).*/\1\/\2/p' <<<"$C3_OUT" | tail -1)"
if [ "$C3_SUM" = "5/5" ]; then C3_OK=1; else C3_OK=0; fi
ck "C3-t7-bytediff-5img" "$C3_OK" \
   "tools:1020 vs dam-ai:8092 byte-diff on 5 fresh prod images: pass=$C3_SUM (features/opqCode/hexColors)"

# ---- C4: T13 semantic + color search online -----------------------------------
cp "$REPO/reports/t13-src/t13_e2e_round3.py" /tmp/t14_t13e2e.py
docker cp /tmp/t14_t13e2e.py "$IFA:/tmp/t13_e2e_round3.py" >/dev/null
C4_E2E="$(timeout 300 docker exec "$IFA" "$SJ_PY" /tmp/t13_e2e_round3.py all 2>&1 | grep -E '"step"|COLOR PASS')"
C4_ES_HITS="$(sed -n 's/.*"step": "es_query".*"hits": \([0-9]*\).*/\1/p' <<<"$C4_E2E" | tail -1)"
C4_TAG="$(grep -c '"provider": "damai_tag", .*"hasCaption": true' <<<"$C4_E2E")"
C4_COLOR="$(sed -n 's/.*COLOR PASS \([0-9]*\)\/\([0-9]*\).*/\1\/\2/p' <<<"$C4_E2E" | tail -1)"
C4_CS="$(timeout 240 docker exec -i "$IFA" "$SJ_PY" - <<'PYEOF' 2>/dev/null | grep -E 'COLORSEARCH' || true
import json, sys
sys.path.insert(0, "/data/projects/image-front-api")
from vcgImageAI.subProjects.imageSearch.service.searchInfoBuilder import SearchInfoBuilder
from vcgImageAI.subProjects.imageSearch.service.esBackendService import EsBackendService
si = SearchInfoBuilder().buildSearchInfo(params={"qColorsInfo": "#ff5733-70,#33ff57-30", "pageSize": 5, "page": 1})
r = EsBackendService().search(searchInfo=si)
hits = len(r.responseItems or [])
print(f"COLORSEARCH codes={bool(si.colorCodes)} hits={hits}")
print("COLORSEARCH PASS" if si.colorCodes and hits > 0 else "COLORSEARCH FAIL")
PYEOF
)"
C4_CS_HITS="$(sed -n 's/.*hits=\([0-9]*\).*/\1/p' <<<"$C4_CS" | tail -1)"
C4_CS_OK="$(grep -c 'COLORSEARCH PASS' <<<"$C4_CS")"
if [ "${C4_ES_HITS:-0}" -gt 0 ] && [ "${C4_TAG:-0}" -ge 1 ] && [ "$C4_COLOR" = "3/3" ] \
   && [ "${C4_CS_HITS:-0}" -gt 0 ] && [ "${C4_CS_OK:-0}" -ge 1 ]; then C4_OK=1; else C4_OK=0; fi
ck "C4-t13-search-online" "$C4_OK" \
   "semantic ES hits=$C4_ES_HITS, damai_tag caption ok=$C4_TAG, color byte-diff=$C4_COLOR, color-search(qColorsInfo→damai-classify→ES) hits=$C4_CS_HITS"

# ---- C5: T10 leak audit (shipped surface, git-tracked files only) --------------
cd "$REPO" || exit 1
ALLOW_LINE_RE='mlib_data/zhaoyufei_cache/soujpg/models|/root/.cache/soujpg/models'
# shipped surface = tracked files minus internal-docs/deploy/reports allowlist dirs
SHIPPED="$(git ls-files | grep -vE '^(reports/|docs/|deploy/|PLAN\.md$|tests/test_embedder_live\.py$|tests/test_degraded\.py$)')"
C5_HITS="$(grep -il soujpg $SHIPPED 2>/dev/null || true)"
# classic/color.py provenance docstrings are accepted (T10 verdict) — count separately
C5_CODE=""
if [ -n "$C5_HITS" ]; then
  while IFS= read -r f; do
    while IFS= read -r ln; do
      C5_CODE+="$(printf '%s:%s' "$f" "$ln"); "
    done < <(grep -in soujpg "$f" | grep -vE "$ALLOW_LINE_RE" | cut -d: -f1 | tr '\n' ' ')
  done <<<"$C5_HITS"
fi
C5_N="$(grep -o ';' <<<"$C5_CODE" | wc -l)"
C5_OK=1; C5_NOTE="shipped surface: 0 non-allowlisted 'soujpg' hits"
if [ "$C5_N" -gt 0 ]; then
  # accepted exception: classic/color.py docstring provenance references (comments only)
  if [ "$(grep -c 'classic/color.py' <<<"$C5_CODE")" -eq "$C5_N" ]; then
    C5_NOTE="accepted: classic/color.py provenance docstrings only (${C5_CODE})"
  else
    C5_OK=0
    C5_NOTE="NON-ALLOWED hits: ${C5_CODE}"
  fi
fi
ck "C5a-t10-leak-soujpg-shipped" "$C5_OK" "$C5_NOTE"

C5_CREDS="$(git grep -inE 'vcgjasstion|mongodb://|password|api[_-]?key *=' -- ':!reports' ':!LICENSE' ':!LICENSES' ':!*/LICENSE*' 2>/dev/null | grep -viE 'openai|api_key.*empty|API key|api-key|PASSWORDS?\b.*env|getenv' | head -5)"
if [ -z "$C5_CREDS" ]; then C5B_OK=1; else C5B_OK=0; fi
ck "C5b-t10-leak-credentials" "$C5B_OK" \
   "tracked files: no credentials/mongo URIs outside reports/ ${C5_CREDS:+hits=$C5_CREDS}"

C5_IP="$(git grep -inE 'vcgjasstion|192\.168\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[01]\.|10\.[0-9]+\.[0-9]+\.|dev\.yufei\.com' -- ':!reports' ':!*.pyc' 2>/dev/null || true)"
C5_IP_BAD=""
while IFS= read -r hit; do
  [ -z "$hit" ] && continue
  f="${hit%%:*}"
  case "$f" in
    deploy/*|docs/*) ;;  # deploy notes + internal design doc examples (T10 allowed range)
    *) C5_IP_BAD+="$hit; " ;;
  esac
done <<<"$C5_IP"
if [ -z "$C5_IP_BAD" ]; then C5C_OK=1; else C5C_OK=0; fi
ck "C5c-t10-leak-internal-endpoints" "$C5C_OK" \
   "internal IPs/hosts confined to deploy/ + docs/ (design doc LB examples) ${C5_IP_BAD:+OUTSIDE=$C5_IP_BAD}"

# ---- summary -------------------------------------------------------------------
emit ""
emit "SUMMARY pass=$PASS fail=$FAIL"
say "verify_all done: PASS=$PASS FAIL=$FAIL (report: $REPORT)"
[ "$FAIL" -eq 0 ]
