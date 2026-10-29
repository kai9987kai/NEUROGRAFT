"""Run or strictly verify the isolated transition-revision study from source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from neurograft.revision_study import DEFAULT_SEEDS, run_revision_study, verify_revision_study
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--out', type=Path)
    target.add_argument('--verify', type=Path)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--window', type=int, choices=(4, 8, 16), default=8)
    parser.add_argument('--budget', type=int, default=64)
    args = parser.parse_args(argv)
    if args.verify:
        result = verify_revision_study(args.verify, replay=args.replay)
    else:
        if args.replay:
            parser.error('--replay requires --verify')
        try:
            seeds = tuple(int(value.strip()) for value in args.seeds.split(','))
        except ValueError:
            parser.error('--seeds must be comma-separated integers')
        report = run_revision_study(args.out, seeds=seeds, window=args.window, budget=args.budget)
        result = {'out': str(args.out.resolve()), 'schema': report['schema'],
                  'seeds': report['protocol']['config']['seeds'], 'promotion_performed': False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
