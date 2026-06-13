import subprocess
import sys


def test_cli_modules_do_not_import_pipeline_eagerly():
    code = (
        "import sys; "
        "import media_search_reliability.cli; "
        "assert 'media_search_reliability.pipeline' not in sys.modules; "
        "assert 'lightgbm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
