import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeVerdict } from '../components/detection/AttackVerdict';

const mk = (cls: string, n: number, conf = 0.99) =>
  Array.from({ length: n }, (_, i) => ({ id: `${cls}-${i}`, predictedClass: cls, confidence: conf,
    isMalicious: cls !== 'Benign', riskLevel: 'high', modelVersion: 't', timestamp: '' } as any));

test('clear leader is named alone', () => {
  const v = computeVerdict([...mk('DDoS-ACK', 990), ...mk('DDoS-SYN', 10), ...mk('Benign', 5)]);
  assert.equal(v.main.length, 1); assert.equal(v.main[0].cls, 'DDoS-ACK'); assert.equal(v.family, 'DDoS');
});

test('near-tie is reported as co-main attacks, family decided first', () => {
  const v = computeVerdict([...mk('DDoS-SYN', 99), ...mk('DDoS-TCP', 97), ...mk('Mirai-udpplain', 22)]);
  assert.deepEqual(v.main.map(r => r.cls), ['DDoS-SYN', 'DDoS-TCP']);
  assert.equal(v.family, 'DDoS'); assert.ok(v.familyClear);
});

test('confidence weighting: many unsure flows lose to fewer confident ones', () => {
  const v = computeVerdict([...mk('Mirai-greip_flood', 100, 0.4), ...mk('Mirai-greeth_flood', 70, 0.99)]);
  assert.equal(v.rows[0].cls, 'Mirai-greeth_flood');
});

test('mostly benign traffic gives no attack', () => {
  assert.equal(computeVerdict([...mk('Benign', 990), ...mk('DDoS-TCP', 10)]).kind, 'none');
});

test('balanced families are reported as mixed', () => {
  const v = computeVerdict([...mk('DDoS-SYN', 100), ...mk('Mirai-udpplain', 90)]);
  assert.equal(v.familyClear, false);
});
