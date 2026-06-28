"""pytest 루트 설정 — 패키지 설치 없이 어느 cwd 에서나 `import kev` 가능하게(P0-4).

기존엔 pyproject/setup/conftest 가 없어 `import kev` 가 repo 루트가 sys.path 에
있을 때(=루트에서 `python -m pytest` 실행)만 동작했다. 이 conftest 가 루트를
sys.path 에 명시적으로 넣어, 타 디렉터리에서 pytest 를 돌려도 import 가 깨지지 않는다.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
