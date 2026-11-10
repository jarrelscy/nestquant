import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kld_alignment import decode_reference_rows, load_tokens, teacher_rows


class AlignmentTest(unittest.TestCase):
    def test_trace_file_gives_tokens_and_prompt_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps(dict(prompt_ids=[5, 6, 7], token_ids=[5, 6, 7, 8, 9])))
            self.assertEqual(load_tokens(path), ([5, 6, 7, 8, 9], 3))

    def test_token_array_has_no_prompt_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'window.npy'
            np.save(path, np.array([[3, 1, 4, 1]], dtype=np.int32))
            self.assertEqual(load_tokens(path), ([3, 1, 4, 1], None))

    def test_rejects_trace_whose_prompt_is_not_a_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps(dict(prompt_ids=[5, 9], token_ids=[5, 6, 7])))
            with self.assertRaises(ValueError):
                load_tokens(path)

    def test_decode_step_rows_align_with_the_same_input_positions(self):
        # Row p of a full prefill is the prediction made after consuming token p.
        # A decode step that consumes token p must be compared with that row.
        reference = np.arange(6 * 2).reshape(6, 2)
        rows = decode_reference_rows(reference, prompt_length=2, steps=3)
        np.testing.assert_array_equal(rows, reference[2:5])

    def test_decode_alignment_rejects_steps_beyond_the_reference(self):
        with self.assertRaises(ValueError):
            decode_reference_rows(np.zeros((4, 2)), prompt_length=2, steps=3)
        with self.assertRaises(ValueError):
            decode_reference_rows(np.zeros((4, 2)), prompt_length=0, steps=1)

    def test_teacher_rows_drop_the_final_unscored_position(self):
        prefill = np.arange(5 * 2).reshape(5, 2)
        np.testing.assert_array_equal(teacher_rows(prefill, teacher_positions=4), prefill[:4])
        with self.assertRaises(ValueError):
            teacher_rows(prefill, teacher_positions=5)


if __name__ == '__main__':
    unittest.main()
