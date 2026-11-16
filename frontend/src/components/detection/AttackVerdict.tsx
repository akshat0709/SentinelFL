import React, { useMemo } from 'react';
import { NetworkFlowRecord } from '@/types/detection';

/** What each of the model's 7 classes means (CIC-IoT-2023 attack definitions). */
export const ATTACK_INFO: Record<string, { family: string; title: string; description: string }> = {
  'DDoS-ACK': {
    family: 'DDoS',
    title: 'DDoS ACK fragmentation flood',
    description: 'Many devices flood the target with fragmented TCP ACK packets, exhausting its processing and reassembly resources.',
  },
  'DDoS-SYN': {
    family: 'DDoS',
    title: 'DDoS SYN flood',
    description: 'TCP connections are opened (SYN) but never completed, filling the target’s connection table so real clients cannot connect.',
  },
  'DDoS-TCP': {
    family: 'DDoS',
    title: 'DDoS TCP flood',
    description: 'A high volume of TCP packets is sent to saturate the target’s bandwidth and network stack.',
  },
  'Mirai-greeth_flood': {
    family: 'Mirai botnet',
    title: 'Mirai GRE-Ethernet flood',
    description: 'Infected IoT devices send GRE-encapsulated Ethernet frames, forcing the target to unpack large volumes of tunnelled traffic.',
  },
  'Mirai-greip_flood': {
    family: 'Mirai botnet',
    title: 'Mirai GRE-IP flood',
    description: 'Infected IoT devices send GRE-encapsulated IP packets to overwhelm the target with tunnelled traffic.',
  },
  'Mirai-udpplain': {
    family: 'Mirai botnet',
    title: 'Mirai plain UDP flood',
    description: 'Infected IoT devices send bare UDP packets with minimal options, maximising packets per second against the target.',
  },
};

const PRIMARY_SHARE = 0.8;      // one attack is >= 80% of malicious flows -> single attack
const MIN_ATTACK_SHARE = 0.05;  // attacks below 5% of malicious flows are listed as minor
const ATTACK_PRESENT = 0.05;    // < 5% malicious flows overall -> no attack

interface Row { cls: string; n: number; share: number; meanConf: number; matched?: number; labelled?: number }

export const AttackVerdict: React.FC<{ flows: NetworkFlowRecord[]; datasetName?: string }> = ({ flows, datasetName }) => {
  const v = useMemo(() => {
    const ok = flows.filter(f => !f.status || f.status === 'classified');
    const total = ok.length;
    const mal = ok.filter(f => f.predictedClass !== 'Benign');
    const by: Record<string, NetworkFlowRecord[]> = {};
    mal.forEach(f => { (by[f.predictedClass] ||= []).push(f); });
    const rows: Row[] = Object.entries(by).map(([cls, fs]) => {
      const labelled = fs.filter(f => f.actualLabel);
      return {
        cls,
        n: fs.length,
        share: fs.length / Math.max(1, mal.length),
        meanConf: fs.reduce((a, f) => a + f.confidence, 0) / fs.length,
        labelled: labelled.length,
        matched: labelled.filter(f => f.actualLabel === cls).length,
      };
    }).sort((a, b) => b.n - a.n);
    const malShare = mal.length / Math.max(1, total);
    const major = rows.filter(r => r.share >= MIN_ATTACK_SHARE);
    const minor = rows.filter(r => r.share < MIN_ATTACK_SHARE);
    let kind: 'none' | 'single' | 'multi' = 'multi';
    if (malShare < ATTACK_PRESENT || rows.length === 0) kind = 'none';
    else if (rows[0].share >= PRIMARY_SHARE) kind = 'single';
    // Ground-truth check: most common true attack label in the file (if labels exist)
    const trueCounts: Record<string, number> = {};
    ok.forEach(f => { if (f.actualLabel && f.actualLabel !== 'Benign') trueCounts[f.actualLabel] = (trueCounts[f.actualLabel] || 0) + 1; });
    const trueTop = Object.entries(trueCounts).sort((a, b) => b[1] - a[1])[0]?.[0];
    return { total, malCount: mal.length, malShare, rows, major, minor, kind, trueTop };
  }, [flows]);

  if (v.total === 0) return null;
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
  const top = v.rows[0];
  const tone =
    v.kind === 'none' ? 'border-forest-600 bg-forest-50' : 'border-red-500 bg-red-50';

  return (
    <div className={`mt-4 rounded-lg border-2 p-4 ${tone}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider font-mono text-charcoal-700">
          Attack Verdict{datasetName ? ` · ${datasetName}` : ''}
        </span>
        <span className="text-[10px] font-mono text-charcoal-500">
          {v.malCount.toLocaleString()} of {v.total.toLocaleString()} flows malicious ({pct(v.malShare)})
        </span>
      </div>

      {v.kind === 'none' ? (
        <p className="mt-1 text-base font-bold text-forest-800">No attack detected: traffic is predominantly benign.</p>
      ) : v.kind === 'single' ? (
        <>
          <p className="mt-1 text-base font-bold text-red-800">
            {ATTACK_INFO[top.cls]?.title || top.cls}{' '}
            <span className="text-sm font-semibold">({ATTACK_INFO[top.cls]?.family || 'attack'})</span>
          </p>
          <p className="text-sm text-charcoal-700">{ATTACK_INFO[top.cls]?.description}</p>
        </>
      ) : (
        <>
          <p className="mt-1 text-base font-bold text-red-800">
            Main attack: {ATTACK_INFO[top.cls]?.title || top.cls}{' '}
            <span className="text-sm font-semibold">({ATTACK_INFO[top.cls]?.family || 'attack'})</span>
          </p>
          <p className="text-sm text-charcoal-700">{ATTACK_INFO[top.cls]?.description}</p>
          <p className="mt-1 text-xs font-semibold text-red-800">
            Most common attack, but only {pct(top.share)} of attack traffic ({top.n.toLocaleString()} of{' '}
            {v.malCount.toLocaleString()} malicious flows): {v.rows.length - 1} other attack type
            {v.rows.length - 1 === 1 ? ' was' : 's were'} also detected (see table).
          </p>
        </>
      )}

      {v.kind !== 'none' && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-charcoal-500">
                <th className="py-1 pr-3">Attack</th>
                <th className="py-1 pr-3">Family</th>
                <th className="py-1 pr-3 text-right">Flows</th>
                <th className="py-1 pr-3 text-right">Share of attack traffic</th>
                <th className="py-1 pr-3 text-right">Mean confidence</th>
                <th className="py-1 text-right">Matches file label</th>
              </tr>
            </thead>
            <tbody>
              {v.rows.map(r => (
                <tr key={r.cls} className={`border-t border-red-200 ${r.share < MIN_ATTACK_SHARE ? 'text-charcoal-400' : 'text-charcoal-900'}`}>
                  <td className="py-1 pr-3 font-semibold" title={ATTACK_INFO[r.cls]?.description}>{ATTACK_INFO[r.cls]?.title || r.cls}</td>
                  <td className="py-1 pr-3">{ATTACK_INFO[r.cls]?.family || '—'}</td>
                  <td className="py-1 pr-3 text-right font-mono">{r.n.toLocaleString()}</td>
                  <td className="py-1 pr-3 text-right font-mono">{pct(r.share)}</td>
                  <td className="py-1 pr-3 text-right font-mono">{pct(r.meanConf)}</td>
                  <td className="py-1 text-right font-mono">{r.labelled ? `${r.matched}/${r.labelled}` : 'no labels'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {v.kind !== 'none' && v.trueTop && v.trueTop !== top.cls && (
        <p className="mt-2 text-xs font-semibold text-amber-800">
          Note: the file’s own labels say the most common attack is {v.trueTop}, not {top.cls}.
        </p>
      )}
      <p className="mt-2 text-[10px] text-charcoal-500">
        Based only on the model’s per-flow predictions. The model knows 6 attack types (above) plus Benign; an attack it was
        never trained on will be reported as the closest of these.
      </p>
    </div>
  );
};
