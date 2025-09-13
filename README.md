# ARB Aggregator — GitHub Actions Pack

목표: `arb_aggregator_v3.py`를 5분마다 실행하고, 실패시 `backfill_collectors.py`가
서버시간(UTC) 기준으로 01~09를 실데이터로 채워 **10개 산출물**을 항상 보장합니다.

## 폴더 구조
/
├─ arb_aggregator_v3.py           # 주니가 올린 원본 집계기(리포 루트에 업로드)
├─ backfill_collectors.py         # 빈/헤더-only 파일을 실데이터로 채움
├─ requirements.txt
└─ .github/workflows/arb_aggregator_backfill.yml

## 실행
1) 위 4개 파일을 리포에 그대로 업로드
2) Actions 탭 → `ARB Aggregator (with Backfill)` → Run workflow
3) 결과: `data/` 폴더 커밋 또는 실행 Artifacts

※ 투자 조언 아님. 거래소 API 정책/가용성에 따라 데이터 변동 가능.