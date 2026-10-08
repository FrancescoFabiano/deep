"""Validation search: greedy best-first search driven through deep's expand server.

deep generates the successors, contracts them (bisimulation, -b), goal-tests
them at generation and detects duplicates with its visited set (-c); this
module only orders the open list by the network's score (higher first, ties
first-in first-out). The search stops when a generated state is a goal
('solved'), after node_cap expansions ('nodecap'), after wall seconds
('timeout'), when deep's resident memory passes cap_gb ('memout') or when the
open list is empty ('exhausted').

deep is started as::

    deep <domain> <problem> --act_lib <lib> -b -c --fast-comparison --ranker_encoding --expand_server <work>/states

in the folder <work>; it writes <work>/fluent_names.csv and goal_ops.csv,
every new state as <work>/states/<id>.dot, and answers each stdin line
'expand <id>' with one line '@@ <id> <action> <is_goal> <is_new>' per
successor and '@@end'.
"""

import csv
import heapq
import itertools
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback

from . import encoding, layout


class Server:
    """One deep process in expand-server mode."""

    def __init__(self, deep, instances, family, problem, work, tier):
        work = os.path.abspath(work)   # deep runs inside work and receives the states folder as a path
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        self.dir = work + '/states'
        cmd = [layout.deep_path(deep), *layout.instance_files(instances, tier, family, problem), '-b', '-c', '--fast-comparison',
               '--ranker_encoding', '--expand_server', self.dir]
        self.p = subprocess.Popen(cmd, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(work + '/deep.err', 'w'), text=True, bufsize=1)
        self.peak_kb = 0
        self._stop = False
        threading.Thread(target=self._watch, daemon=True).start()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('deep exited: ' + open(work + '/deep.err').read()[-500:])
            if line.startswith('@@ready'):
                self.root_goal = line.split()[2] == '1'
                break
        self.names = encoding.read_names(work + '/fluent_names.csv')
        self.ops = encoding.read_ops(work + '/goal_ops.csv')

    def _watch(self):
        while not self._stop and self.p.poll() is None:
            try:
                rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(self.p.pid)], capture_output=True, text=True).stdout
                self.peak_kb = max(self.peak_kb, int(rss.strip() or 0))
            except Exception:
                pass
            time.sleep(0.5)

    def dot(self, sid):
        return f'{self.dir}/{sid}.dot'

    def expand(self, sid):
        """-> [(state id, action, is_goal, is_new)] of the successors of sid."""
        self.p.stdin.write(f'expand {sid}\n')
        self.p.stdin.flush()
        out = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('deep exited')
            if line.startswith('@@end'):
                return out
            if line.startswith('@@ '):
                _, i, a, g, n = line.split()
                out.append((int(i), a, g == '1', n == '1'))

    def cpu(self):
        try:
            t = subprocess.run(['ps', '-o', 'cputime=', '-p', str(self.p.pid)], capture_output=True,
                               text=True).stdout.strip().replace(',', '.')
            return sum(float(x) * 60 ** i for i, x in enumerate(reversed(t.split(':'))))
        except Exception:
            return float('nan')

    def close(self):
        self._stop = True
        try:
            self.p.stdin.write('quit\n')
            self.p.stdin.flush()
            self.p.wait(10)
        except Exception:
            self.p.kill()


def net_scorer(net, srv):
    """Scores states of this server with the network; agent ranks come from the initial state."""
    import torch
    from torch_geometric.data import Batch
    agents = encoding.agent_ranks([srv.dot(0)])
    v0 = len(encoding.vocab)
    net.eval()

    def g(s):
        x, ei, ea, pt = encoding.graph(srv.dot(s), srv.names, agents, srv.ops if getattr(net, 'ops', False) else None)
        if getattr(net, 'anon', False):
            x = encoding.anon_map()[x]
        return encoding.data((x, ei, ea, pt))

    def score(sids):
        with torch.no_grad():
            return net(Batch.from_data_list([g(s) for s in sids])).tolist()

    score.new_types = lambda: len(encoding.vocab) - v0
    return score


def search(deep, instances, family, tier, problem, net, node_cap=None, wall=600.0, cap_gb=6.0, work_root=None, tag='net'):
    """One greedy best-first search; -> dict(status, nodes, plan_length, wall_s, deep_cpu_s, peak_rss_mb, model_s, ...)."""
    work_root = work_root or os.path.join(tempfile.gettempdir(), 'gnn-epddl-search')
    work = f'{work_root}/{problem}-{tag}'
    cap_kb = cap_gb * 2 ** 20
    t0 = time.time()
    srv = Server(deep, instances, family, problem, work, tier)
    mt = 0.0
    res = dict(family=family, problem=problem, config='net', status='', nodes=0, plan_length='', wall_s=0, deep_cpu_s=0,
               peak_rss_mb=0, model_s=0, new_types=0)
    try:
        score = net_scorer(net, srv)
        tie = itertools.count()
        depth = {0: 0}
        openl = [(0.0, next(tie), 0)]
        while openl:
            if time.time() - t0 > wall:
                res['status'] = 'timeout'
                break
            if node_cap and res['nodes'] >= node_cap:
                res['status'] = 'nodecap'
                break
            if srv.peak_kb > cap_kb:
                res['status'] = 'memout'
                break
            cur = heapq.heappop(openl)[2]
            res['nodes'] += 1
            kids = srv.expand(cur)
            if any(g for s, a, g, n in kids):
                res['status'] = 'solved'
                res['plan_length'] = depth[cur] + 1
                break
            new = [s for s, a, g, n in kids if n]
            for s in new:
                depth[s] = depth[cur] + 1
            if new:
                t1 = time.time()
                sc = score(new)
                mt += time.time() - t1
                for s, v in zip(new, sc):
                    heapq.heappush(openl, (-v, next(tie), s))
        else:
            res['status'] = 'exhausted'
        res['new_types'] = score.new_types()
    except Exception as e:
        res['status'] = 'error:' + type(e).__name__
        traceback.print_exc()
    res.update(wall_s=round(time.time() - t0, 1), deep_cpu_s=round(srv.cpu(), 1), peak_rss_mb=round(srv.peak_kb / 1024),
               model_s=round(mt, 1))
    srv.close()
    shutil.rmtree(f'{work}/states', ignore_errors=True)
    return res


def write_row(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
