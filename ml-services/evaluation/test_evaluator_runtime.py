import os
import unittest
from unittest import mock

import evaluator_runtime


class EvaluatorRuntimeTests(unittest.TestCase):
    def test_v1_is_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                evaluator_runtime.resolve_evaluator_version(),
                "v1",
            )

    def test_explicit_version_overrides_environment(self):
        with mock.patch.dict(os.environ, {"EVALUATOR_VERSION": "v2"}):
            self.assertEqual(
                evaluator_runtime.resolve_evaluator_version("V1"),
                "v1",
            )

    def test_unknown_version_is_rejected(self):
        with self.assertRaises(
            evaluator_runtime.EvaluatorConfigurationError
        ):
            evaluator_runtime.resolve_evaluator_version("experimental")

    def test_reserved_v2_is_rejected_until_implemented(self):
        with self.assertRaises(evaluator_runtime.EvaluatorUnavailableError):
            evaluator_runtime.resolve_evaluator_version("v2")

    def test_evaluation_records_version_provenance(self):
        implementation = mock.Mock()
        implementation.evaluate_call.return_value = {
            "rubric_version": "0.4.1"
        }
        with mock.patch(
            "evaluator_runtime.importlib.import_module",
            return_value=implementation,
        ):
            result = evaluator_runtime.evaluate_call("call-1")

        self.assertEqual(result["_evaluator"]["version"], "v1")
        self.assertEqual(result["_evaluator"]["implementation"], "graph")
        implementation.evaluate_call.assert_called_once_with(
            "call-1",
            results_dir="results",
        )


if __name__ == "__main__":
    unittest.main()
