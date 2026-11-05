#!/usr/bin/env python3
"""Thin wrapper: python scripts/init_db.py"""
from wde.cli import main

if __name__ == "__main__":
    main(["init-db"], standalone_mode=True)
