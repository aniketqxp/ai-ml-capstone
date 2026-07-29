import LegacyCallDetail from './LegacyCallDetail';
import EvaluatorV2CallDetail from './EvaluatorV2CallDetail';

const V2_ENABLED = ['1', 'true', 'yes'].includes(
  String(import.meta.env.VITE_EVALUATOR_V2 ?? '').toLowerCase(),
);

export default function CallDetail() {
  return V2_ENABLED ? <EvaluatorV2CallDetail /> : <LegacyCallDetail />;
}
