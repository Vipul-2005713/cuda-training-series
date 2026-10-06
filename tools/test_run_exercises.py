"""CPU-only regression checks: python3 -m unittest discover -s tools -p 'test_*.py'."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_exercises as runner


class RunnerTests(unittest.TestCase):
    def test_wsl_driver_takes_precedence_without_losing_other_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = Path(directory)
            (driver / 'libcuda.so.1').touch()
            with patch.object(runner, 'WSL_DRIVER_DIR', driver), \
                    patch.object(runner.platform, 'release', return_value='6.6-microsoft-standard-WSL2'), \
                    patch.dict(os.environ, {'LD_LIBRARY_PATH': f'/custom/cuda:{driver}'}):
                env = runner.runtime_environment()
                self.assertEqual(env['LD_LIBRARY_PATH'], f'{driver}:/custom/cuda')
                self.assertEqual(os.environ['LD_LIBRARY_PATH'], f'/custom/cuda:{driver}')
            with patch.object(runner.platform, 'release', return_value='6.8.0-generic'), \
                    patch.dict(os.environ, {'LD_LIBRARY_PATH': '/custom/cuda'}):
                self.assertEqual(runner.runtime_environment()['LD_LIBRARY_PATH'], '/custom/cuda')

    def call_main(self, *arguments):
        output = io.StringIO()
        with patch('sys.argv', ['run_exercises.py', *arguments]), contextlib.redirect_stdout(output):
            result = runner.main()
        return result, output.getvalue()

    def test_homework_number_does_not_select_hw10(self):
        result, output = self.call_main('--hw', '1', '--list')
        self.assertEqual(result, 0)
        self.assertEqual(len(output.splitlines()), 3)
        self.assertNotIn('hw10_', output)
        self.assertNotIn('_solution', output)

    def test_solutions_receive_the_same_boundary_inputs(self):
        cases = list(runner.cases('edge'))
        original = [(case, args) for name, case, args in cases if name == 'hw9_compaction']
        reference = [(case, args) for name, case, args in cases if name == 'hw9_compaction_solution']
        self.assertTrue(original)
        self.assertEqual(original, reference)

    def test_invalid_selection_is_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.call_main('--only', 'does_not_exist', '--list')
        self.assertEqual(error.exception.code, 2)

    def test_failed_build_never_executes_stale_binary(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            stale = Path(directory) / 'build/ubuntu/hw1_hello'
            stale.parent.mkdir(parents=True)
            stale.write_text('stale executable')
            with patch.object(runner.shutil, 'which', return_value='/usr/bin/nvcc'), \
                    patch.object(runner, 'invoke', return_value=(1, 'compile failure', 0.1)) as invoke:
                result, _ = self.call_main('--build', '--only', 'hw1_hello')
            self.assertEqual(result, 1)
            self.assertEqual(invoke.call_count, 1)  # Only the compiler; no stale program.
            summary = json.loads((Path(directory) / 'results/ubuntu/summary.json').read_text())
            self.assertEqual(summary['builds'][0]['status'], 'FAIL')
            self.assertEqual(summary['runs'][0]['exit_code'], 125)

    def test_current_results_replace_history_and_allow_external_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / 'repo', Path(directory) / 'external results'
            root.mkdir()
            output.mkdir()
            (output / 'summary.json').write_text('{"runs": [{"status": "FAIL"}]}')
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner, 'invoke', return_value=(0, 'MULTI_GPU_LIMITATION\nPASS', 0.1)):
                result, _ = self.call_main('--only', 'hw10_openmp', '--output', str(output))
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(result, 0)
            self.assertEqual(len(summary['runs']), 1)
            self.assertEqual(summary['runs'][0]['status'], 'PASS_WITH_SKIPS')

    def test_runtime_failure_remains_failure_even_if_output_mentions_skip(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), \
                patch.object(runner, 'invoke', return_value=(1, 'SKIP feature\nFAIL validation', 0.1)):
            result, _ = self.call_main('--only', 'hw6_array_inc')
            summary = json.loads((Path(directory) / 'results/ubuntu/summary.json').read_text())
            self.assertEqual(result, 1)
            self.assertEqual(summary['runs'][0]['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
