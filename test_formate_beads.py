"""
Test Suite for Formate Beads Experiment Designer
================================================
Tests the core functionality of the MPC-enhanced substrate control system.

Run with: pytest test_formate_beads.py -v
Or: python test_formate_beads.py
"""

import unittest
import numpy as np
import sys

# Import the necessary classes from the GUI file
# Note: This assumes the classes are defined in formate_beads_gui.py
try:
    from formate_beads_gui import (
        Bead, ExperimentManager, MonodKinetics, ConstantSubstrateCalculator,
        M07_BEAD_RELEASE, M03_BEAD_RELEASE, M07_FORMATE_CONTENT, 
        M03_FORMATE_CONTENT, FORMATE_MW, MU_MAX, K_S, Y_XS
    )
except ImportError as e:
    print(f"Error importing from formate_beads_gui.py: {e}")
    print("Make sure formate_beads_gui.py is in the same directory")
    sys.exit(1)


class TestBeadReleaseProfiles(unittest.TestCase):
    """Test bead release profile data integrity."""
    
    def test_m07_release_profile_exists(self):
        """M07 release profile should have data."""
        self.assertIsNotNone(M07_BEAD_RELEASE)
        self.assertGreater(len(M07_BEAD_RELEASE), 0)
    
    def test_m03_release_profile_exists(self):
        """M03 release profile should have data."""
        self.assertIsNotNone(M03_BEAD_RELEASE)
        self.assertGreater(len(M03_BEAD_RELEASE), 0)
    
    def test_release_rates_are_positive(self):
        """All release rates should be non-negative."""
        for rate in M07_BEAD_RELEASE.values():
            self.assertGreaterEqual(rate, 0, "M07 release rates should be non-negative")
        
        for rate in M03_BEAD_RELEASE.values():
            self.assertGreaterEqual(rate, 0, "M03 release rates should be non-negative")
    
    def test_release_profiles_start_at_day_1(self):
        """Release profiles should start at day 1."""
        self.assertIn(1, M07_BEAD_RELEASE)
        self.assertIn(1, M03_BEAD_RELEASE)
    
    def test_m07_has_higher_initial_release(self):
        """M07 should have higher release on day 1 than M03."""
        self.assertGreater(M07_BEAD_RELEASE[1], M03_BEAD_RELEASE[1])


class TestBeadClass(unittest.TestCase):
    """Test the Bead class functionality."""
    
    def test_create_m07_bead(self):
        """Should create M07 bead successfully."""
        bead = Bead('M07', day_added=0)
        self.assertEqual(bead.bead_type, 'M07')
        self.assertEqual(bead.day_added, 0)
    
    def test_create_m03_bead(self):
        """Should create M03 bead successfully."""
        bead = Bead('M03', day_added=0)
        self.assertEqual(bead.bead_type, 'M03')
        self.assertEqual(bead.day_added, 0)
    
    def test_bead_release_before_addition(self):
        """Bead should not release before it's added."""
        bead = Bead('M07', day_added=2.0)
        release = bead.get_release_rate(1.0)
        self.assertEqual(release, 0.0)
    
    def test_bead_release_on_first_day(self):
        """Bead should release on first day after addition."""
        bead = Bead('M07', day_added=0)
        release = bead.get_release_rate(1.0)
        self.assertGreater(release, 0)
    
    def test_bead_release_decreases_over_time(self):
        """Bead release should generally decrease over time."""
        bead = Bead('M07', day_added=0)
        release_day1 = bead.get_release_rate(1.0)
        release_day3 = bead.get_release_rate(3.0)
        # M07 should have higher release on day 1 than day 3
        self.assertGreater(release_day1, release_day3)
    
    def test_bead_release_stops_after_max_day(self):
        """Bead should stop releasing after maximum day."""
        bead = Bead('M07', day_added=0)
        max_day = max(M07_BEAD_RELEASE.keys())
        release = bead.get_release_rate(max_day + 2)
        self.assertEqual(release, 0.0)


class TestBeadManager(unittest.TestCase):
    """Test the ExperimentManager class."""
    
    def test_create_empty_manager(self):
        """Should create empty experiment manager."""
        manager = ExperimentManager()
        self.assertEqual(len(manager.beads), 0)
    
    def test_add_m07_bead(self):
        """Should add M07 bead to manager."""
        manager = ExperimentManager()
        manager.add_bead('M07', 0)
        self.assertEqual(len(manager.beads), 1)
    
    def test_add_multiple_beads(self):
        """Should add multiple beads."""
        manager = ExperimentManager()
        manager.add_bead('M07', 0)
        manager.add_bead('M03', 0)
        manager.add_bead('M07', 1.0)
        self.assertEqual(len(manager.beads), 3)
    
    def test_total_release_single_bead(self):
        """Total release should match single bead release."""
        manager = ExperimentManager()
        manager.add_bead('M07', 0)
        
        bead = Bead('M07', 0)
        expected_release = bead.get_release_rate(1.0)
        actual_release = manager.get_total_release(1.0)
        
        self.assertAlmostEqual(actual_release, expected_release, places=6)
    
    def test_total_release_multiple_beads(self):
        """Total release should sum multiple beads."""
        manager = ExperimentManager()
        manager.add_bead('M07', 0)
        manager.add_bead('M03', 0)
        
        bead1 = Bead('M07', 0)
        bead2 = Bead('M03', 0)
        expected_release = bead1.get_release_rate(1.0) + bead2.get_release_rate(1.0)
        actual_release = manager.get_total_release(1.0)
        
        self.assertAlmostEqual(actual_release, expected_release, places=6)


class TestMonodKinetics(unittest.TestCase):
    """Test Monod kinetics calculations."""
    
    def setUp(self):
        """Set up test parameters."""
        self.monod = MonodKinetics(mu_max=MU_MAX, K_s=K_S, Y_xs=Y_XS)
    
    def test_growth_rate_zero_substrate(self):
        """Growth rate should be zero with no substrate."""
        mu = self.monod.growth_rate(0)
        self.assertEqual(mu, 0.0)
    
    def test_growth_rate_high_substrate(self):
        """Growth rate should approach mu_max at high substrate."""
        # At S = 10 * K_s, growth should be ~91% of mu_max
        substrate = 10 * K_S
        mu = self.monod.growth_rate(substrate)
        expected = MU_MAX * substrate / (K_S + substrate)
        self.assertAlmostEqual(mu, expected, places=6)
    
    def test_growth_rate_at_half_saturation(self):
        """Growth rate should be half mu_max at K_s."""
        mu = self.monod.growth_rate(K_S)
        self.assertAlmostEqual(mu, MU_MAX / 2, places=6)
    
    def test_consumption_rate_zero_od(self):
        """Consumption should be zero with no bacteria."""
        rate = self.monod.consumption_rate(30.0, 0.0)
        self.assertEqual(rate, 0.0)
    
    def test_consumption_rate_increases_with_od(self):
        """Consumption should increase with OD."""
        rate1 = self.monod.consumption_rate(30.0, 0.01)
        rate2 = self.monod.consumption_rate(30.0, 0.02)
        self.assertGreater(rate2, rate1)
    
    def test_consumption_rate_increases_with_substrate(self):
        """Consumption should increase with substrate (up to saturation)."""
        rate1 = self.monod.consumption_rate(10.0, 0.02)
        rate2 = self.monod.consumption_rate(30.0, 0.02)
        self.assertGreater(rate2, rate1)


class TestConstantSubstrateCalculator(unittest.TestCase):
    """Test the main calculator class."""
    
    def setUp(self):
        """Set up test calculator."""
        self.calculator = ConstantSubstrateCalculator(
            volume=0.1,
            monod_params={'mu_max': MU_MAX, 'K_s': K_S, 'Y_xs': Y_XS},
            target_concentration=30.0
        )
    
    def test_calculator_initialization(self):
        """Calculator should initialize with correct parameters."""
        self.assertEqual(self.calculator.volume, 0.1)
        self.assertEqual(self.calculator.target_concentration, 30.0)
    
    def test_calculate_bead_schedule_basic(self):
        """Should calculate bead schedule without errors."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=3,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        # Check that results contain expected keys
        self.assertIn('times', results)
        self.assertIn('substrate', results)
        self.assertIn('od', results)
        self.assertIn('bead_schedule', results)
        self.assertIn('release_rates', results)
        self.assertIn('consumption_rates', results)
    
    def test_initial_beads_added(self):
        """Should add initial beads at time 0."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=3,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        self.assertIn(0, results['bead_schedule'])
        initial = results['bead_schedule'][0]
        self.assertGreater(initial['M07'] + initial['M03'], 0)
    
    def test_od_increases_over_time(self):
        """Bacterial OD should increase over time."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=5,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        self.assertGreater(results['od'][-1], results['od'][0])
    
    def test_substrate_stays_near_target(self):
        """Substrate should stay reasonably close to target."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=5,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        # After initial stabilization, substrate should be near target
        avg_substrate = np.mean(results['substrate'][len(results['substrate'])//2:])
        # Should be within 20% of target on average
        self.assertGreater(avg_substrate, self.calculator.target_concentration * 0.8)
        self.assertLess(avg_substrate, self.calculator.target_concentration * 1.2)
    
    def test_cumulative_consumption_increases(self):
        """Cumulative consumption should monotonically increase."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=3,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        cumulative = results['cumulative_consumed']
        for i in range(1, len(cumulative)):
            self.assertGreaterEqual(cumulative[i], cumulative[i-1])
    
    def test_hcl_matches_consumption(self):
        """HCl needed should match cumulative consumption (1:1 ratio)."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=3,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        # Total HCl should equal total consumption
        total_hcl = results['hcl_needed_cumulative'][-1]
        total_consumption = results['cumulative_consumed'][-1]
        self.assertAlmostEqual(total_hcl, total_consumption, places=6)


class TestInterventionIntervals(unittest.TestCase):
    """Test different intervention interval scenarios."""
    
    def setUp(self):
        """Set up calculator for testing."""
        self.calculator = ConstantSubstrateCalculator(
            volume=0.1,
            monod_params={'mu_max': MU_MAX, 'K_s': K_S, 'Y_xs': Y_XS},
            target_concentration=30.0
        )
    
    def test_daily_intervention(self):
        """Should handle daily (1.0 day) intervention."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=3,
            dt=0.05,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        # Should have interventions near 0, 1, 2, 3 days
        schedule_days = sorted(results['bead_schedule'].keys())
        self.assertIn(0, schedule_days)
    
    def test_twice_daily_intervention(self):
        """Should handle twice-daily (0.5 day) intervention."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=2,
            dt=0.05,
            intervention_interval=0.5,
            lower_threshold=0.90
        )
        
        # Should have interventions at smaller intervals
        schedule_days = sorted(results['bead_schedule'].keys())
        self.assertIn(0, schedule_days)
    
    def test_longer_interval(self):
        """Should handle longer (2.0 day) intervention intervals."""
        results = self.calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=6,
            dt=0.1,
            intervention_interval=2.0,
            lower_threshold=0.85
        )
        
        schedule_days = sorted(results['bead_schedule'].keys())
        self.assertIn(0, schedule_days)
        # With 2-day intervals, should have fewer interventions
        self.assertLessEqual(len(schedule_days), 5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_very_small_volume(self):
        """Should handle very small volumes."""
        calculator = ConstantSubstrateCalculator(
            volume=0.01,  # 10 mL
            monod_params={'mu_max': MU_MAX, 'K_s': K_S, 'Y_xs': Y_XS},
            target_concentration=20.0
        )
        
        results = calculator.calculate_bead_schedule(
            initial_od=0.01,
            experiment_days=3,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        self.assertIsNotNone(results)
        self.assertGreater(len(results['bead_schedule']), 0)
    
    def test_high_initial_od(self):
        """Should handle high initial OD."""
        calculator = ConstantSubstrateCalculator(
            volume=0.1,
            monod_params={'mu_max': MU_MAX, 'K_s': K_S, 'Y_xs': Y_XS},
            target_concentration=30.0
        )
        
        results = calculator.calculate_bead_schedule(
            initial_od=0.1,  # High OD
            experiment_days=3,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        # Should add more beads initially due to higher consumption
        initial_beads = results['bead_schedule'][0]
        total_initial = initial_beads['M07'] + initial_beads['M03']
        self.assertGreater(total_initial, 5)  # Should need more beads
    
    def test_low_threshold(self):
        """Should handle low action threshold."""
        calculator = ConstantSubstrateCalculator(
            volume=0.1,
            monod_params={'mu_max': MU_MAX, 'K_s': K_S, 'Y_xs': Y_XS},
            target_concentration=30.0
        )
        
        results = calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=3,
            dt=0.1,
            intervention_interval=1.0,
            lower_threshold=0.80  # 80% - relaxed control
        )
        
        # Should still complete successfully
        self.assertIsNotNone(results)
        self.assertGreater(results['od'][-1], results['od'][0])


class TestMassBalance(unittest.TestCase):
    """Test mass balance and conservation principles."""
    
    def test_yield_coefficient_consistency(self):
        """OD increase should match substrate consumption via yield coefficient."""
        calculator = ConstantSubstrateCalculator(
            volume=0.1,
            monod_params={'mu_max': MU_MAX, 'K_s': K_S, 'Y_xs': Y_XS},
            target_concentration=30.0
        )
        
        results = calculator.calculate_bead_schedule(
            initial_od=0.02,
            experiment_days=5,
            dt=0.01,
            intervention_interval=1.0,
            lower_threshold=0.95
        )
        
        # Calculate expected OD from consumption
        total_consumed_per_L = results['cumulative_consumed'][-1] / calculator.volume
        expected_od_increase = total_consumed_per_L * Y_XS
        actual_od_increase = results['od'][-1] - results['od'][0]
        
        # Should match within 1% (accounting for numerical integration)
        relative_error = abs(expected_od_increase - actual_od_increase) / actual_od_increase
        self.assertLess(relative_error, 0.01, f"Mass balance error: {relative_error*100:.2f}%")


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBeadReleaseProfiles))
    suite.addTests(loader.loadTestsFromTestCase(TestBeadClass))
    suite.addTests(loader.loadTestsFromTestCase(TestBeadManager))
    suite.addTests(loader.loadTestsFromTestCase(TestMonodKinetics))
    suite.addTests(loader.loadTestsFromTestCase(TestConstantSubstrateCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestInterventionIntervals))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestMassBalance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when executed directly
    success = run_tests()
    sys.exit(0 if success else 1)
