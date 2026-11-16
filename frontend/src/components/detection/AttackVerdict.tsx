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

const ATTACK_PRESENT = 0.05;   // < 5% of flows predicted malicious -> no attack
const CLEAR_LEAD = 1.5;        // a leader must have >= 1.5x the votes of the runner-up
const MAX_CO_MAIN = 3;         // name at most 3 co-main attacks

interface Row {
  cls: string; n: number; votes: number; share: number; meanConf: number;
  matched: number; labelled: number; isMain: boolean;
}

/**
 * Dataset verdict from per-flow predictions:
 *  1. votes = sum of each flow's confidence (confidence-weighted, so unsure flows count less)
 *  2. family first (DDoS vs Mirai botnet): robust to the model's main error, greeth <-> greip
 *  3. a specific attack is named alone only if it has >= 1.5x the votes of the next one;
 *     otherwise every attack within that margin of the leader is named as co-main.
 */
export function computeVerdict(flows: NetworkFlowRecord[]) {
  const ok = flows.filter(f => !f.status || f.status === 'classified');
  const total = ok.length;
  const mal = ok.filter(f => f.predictedClass !== 'Benign');
  const by: Record<string, NetworkFlowRecord[]> = {};
  mal.forEach(f => { (by[f.predictedClass] ||= []).push(f); });
  const totalVotes = mal.reduce((a, f) => a + f.confidence, 0);

  const rows: Row[] = Object.entries(by).map(([cls, fs]) => {
    const votes = fs.reduce((a, f) => a + f.confidence, 0);
    const labelled = fs.filter(f => f.actualLabel);
    return {
      cls, n: fs.length, votes,
      share: votes / Math.max(1e-9, totalVotes),
      meanConf: votes / fs.length,
      labelled: labelled.length,
      matched: labelled.filter(f => f.actualLabel === cls).length,
      isMain: false,
    };
  }).sort((a, b) => b.votes - a.votes);

  const malShare = mal.length / Math.max(1, total);
  const kind: 'none' | 'attack' = malShare < ATTACK_PRESENT || rows.length === 0 ? 'none' : 'attack';

  // Family vote
  const famVotes: Record<string, number> = {};
  rows.forEach(r => {
    const fam = ATTACK_INFO[r.cls]?.family || 'Other';
    famVotes[fam] = (famVotes[fam] || 0) + r.votes;
  });
  const fams = Object.entries(famVotes).sort((a, b) => b[1] - a[1]);
  const familyClear = fams.length === 1 || (fams[1] && fams[0][1] >= CLEAR_LEAD * fams[1][1]);
  const family = familyClear ? fams[0]?.[0] : fams.map(f => f[0]).join(' + ');
  const familyShare = fams[0] ? fams[0][1] / Math.max(1e-9, totalVotes) : 0;

  // Main attack(s): everything within the clear-lead margin of the leader
  const main: Row[] = [];
  if (rows.length) {
    const lead = rows[0].votes;
    for (const r of rows) {
      if (main.length < MAX_CO_MAIN && r.votes * CLEAR_LEAD > lead) { r.isMain = true; main.push(r); }
    }
  }

  const trueCounts: Record<string, number> = {};
  ok.forEach(f => { if (f.actualLabel && f.actualLabel !== 'Benign') trueCounts[f.actualLabel] = (trueCounts[f.actualLabel] || 0) + 1; });
  const trueTop = Object.entries(trueCounts).sort((a, b) => b[1] - a[1])[0]?.[0];

  return { total, malCount: mal.length, malShare, rows, kind, family, familyClear, familyShare, fams, main, trueTop };
}

export const AttackVerdict: React.FC<{ flows: NetworkFlowRecord[]; datasetName?: string }> = ({ flows, datasetName }) => {
  const v = useMemo(() => computeVerdict(flows), [flows]);
  if (v.total === 0) return null;

  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
  const conf = (c: number) => (c >= 0.99995 ? '>99.99%' : `${(c * 100).toFixed(c >= 0.9995 ? 2 : 1)}%`);
  const name = (cls: string) => ATTACK_INFO[cls]?.title || cls;
  const single = v.main.length === 1 ? v.main[0] : null;
  const tone = v.kind === 'none' ? 'border-forest-600 bg-forest-50' : 'border-red-500 bg-red-50';

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
      ) : (
        <>
          <p className="mt-1 text-xs font-semibold uppercase tracking-wide text-red-700">
            {v.familyClear
              ? `${v.family} attack (${pct(v.familyShare)} of attack votes)`
              : `Mixed attack families: ${v.fams.map(([f, n]) => `${f} ${pct(n / Math.max(1e-9, v.fams.reduce((a, x) => a + x[1], 0)))}`).join(' · ')}`}
          </p>
          {single ? (
            <>
              <p className="text-base font-bold text-red-800">Main attack: {name(single.cls)}</p>
              <p className="text-sm text-charcoal-700">{ATTACK_INFO[single.cls]?.description}</p>
              {v.rows.length > 1 && (
                <p className="mt-1 text-xs text-red-800">
                  {pct(single.share)} of attack votes, at least {CLEAR_LEAD}× the next attack; {v.rows.length - 1} other
                  type{v.rows.length - 1 === 1 ? '' : 's'} also seen (table).
                </p>
              )}
            </>
          ) : (
            <>
              <p className="text-base font-bold text-red-800">
                Main attacks (too close to separate): {v.main.map(r => name(r.cls)).join(' + ')}
              </p>
              <p className="mt-1 text-xs text-red-800">
                {v.main.map(r => `${name(r.cls)} ${pct(r.share)}`).join(' vs ')} of attack votes: none leads the next by{' '}
                {CLEAR_LEAD}×, so they are reported together.
              </p>
            </>
          )}

          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-charcoal-500">
                  <th className="py-1 pr-3">Attack</th>
                  <th className="py-1 pr-3">Family</th>
                  <th className="py-1 pr-3 text-right">Flows</th>
                  <th className="py-1 pr-3 text-right" title="Sum of per-flow confidence">Votes</th>
                  <th className="py-1 pr-3 text-right">Share of votes</th>
                  <th className="py-1 pr-3 text-right">Mean confidence</th>
                  <th className="py-1 text-right">Matches file label</th>
                </tr>
              </thead>
              <tbody>
                {v.rows.map(r => (
                  <tr key={r.cls} className={`border-t border-red-200 ${r.isMain ? 'font-semibold text-charcoal-900' : 'text-charcoal-500'}`}>
                    <td className="py-1 pr-3" title={ATTACK_INFO[r.cls]?.description}>
                      {name(r.cls)}{r.isMain && <span className="ml-1 text-red-700">(main)</span>}
                    </td>
                    <td className="py-1 pr-3">{ATTACK_INFO[r.cls]?.family || '—'}</td>
                    <td className="py-1 pr-3 text-right font-mono">{r.n.toLocaleString()}</td>
                    <td className="py-1 pr-3 text-right font-mono">{r.votes.toFixed(1)}</td>
                    <td className="py-1 pr-3 text-right font-mono">{pct(r.share)}</td>
                    <td className="py-1 pr-3 text-right font-mono">{conf(r.meanConf)}</td>
                    <td className="py-1 text-right font-mono">{r.labelled ? `${r.matched}/${r.labelled}` : 'no labels'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {v.trueTop && !v.main.some(r => r.cls === v.trueTop) && (
            <p className="mt-2 text-xs font-semibold text-amber-800">
              Note: the file’s own labels say the most common attack is {v.trueTop}, which is not among the main attacks above.
            </p>
          )}
        </>
      )}
      <p className="mt-2 text-[10px] text-charcoal-500">
        Votes = sum of per-flow confidence. Family is decided first; a single attack is named only if it has at least{' '}
        {CLEAR_LEAD}× the votes of the next. The model knows 6 attack types plus Benign; an attack it was never trained on is
        reported as the closest of these.
      </p>
    </div>
  );
};
