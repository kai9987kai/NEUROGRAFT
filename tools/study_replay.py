"""Source-checkout wrapper for the matched empirical replay study."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from neurograft.replay_study import (DEFAULT_BUDGETS, DEFAULT_SEEDS,
                                        run_replay_study, verify_replay_study)
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--out', help='new output directory; existing evidence is never overwritten')
    operation.add_argument('--verify', metavar='DIRECTORY')
    parser.add_argument('--replay', action='store_true', help='recompute all deterministic schedule results')
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--budgets', default=','.join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument('--discount', type=float, default=.9)
    args = parser.parse_args()
    try:
        if args.verify:
            result = verify_replay_study(args.verify, replay=args.replay)
        else:
            if args.replay:
                parser.error('--replay requires --verify')
            report = run_replay_study(args.out, seeds=tuple(int(v) for v in args.seeds.split(',')),
                                      budgets=tuple(int(v) for v in args.budgets.split(',')), discount=args.discount)
            result = {'schema': report['schema'], 'instances': len(report['instances']),
                      'out': str(Path(args.out).resolve()), 'promotion_performed': False}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
