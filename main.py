"""Entry point: ``python main.py <command> ...`` (see ``python main.py --help``)."""

import sys

from app.cli.commands import main

if __name__ == "__main__":
    sys.exit(main())
