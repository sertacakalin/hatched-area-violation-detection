import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "plate" / "tr_plate.py"
SPEC = spec_from_file_location("tr_plate", MODULE_PATH)
tr_plate = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tr_plate)

repair_tr_plate = tr_plate.repair_tr_plate
validate_tr_plate = tr_plate.validate_tr_plate


class TrPlateRepairTest(unittest.TestCase):
    def test_keeps_valid_plate(self):
        self.assertEqual(repair_tr_plate("34ABC123"), "34ABC123")

    def test_repairs_digit_positions(self):
        self.assertEqual(repair_tr_plate("O6ABCI23"), "06ABC123")

    def test_repairs_letter_positions(self):
        self.assertEqual(repair_tr_plate("34A8C123"), "34ABC123")

    def test_repaired_plate_validates(self):
        repaired = repair_tr_plate("41USG0")
        is_valid, city_code, city_name = validate_tr_plate(repaired)
        self.assertTrue(is_valid)
        self.assertEqual(city_code, "41")
        self.assertEqual(city_name, "Kocaeli")

    def test_returns_cleaned_text_when_no_valid_candidate(self):
        self.assertEqual(repair_tr_plate("XX???"), "XX")


if __name__ == "__main__":
    unittest.main()
