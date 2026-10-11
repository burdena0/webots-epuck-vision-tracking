"""Set project controllers to this interpreter, without changing global Webots settings."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    for name in ('camera_view', 'benchmark_supervisor', 'audit_supervisor'):
        folder = ROOT/'controllers'/name
        if folder.is_dir():
            (folder/'runtime.ini').write_text('[python]\nCOMMAND = '+sys.executable+'\n', encoding='utf-8')
    print('Configured project controllers for '+sys.executable)

if __name__ == '__main__':
    main()
