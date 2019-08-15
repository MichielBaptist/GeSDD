import random
import utils.string_utils as stru

class indicator_manager:

    def __init__(self, regions):
        # Regions: A collection of range(low, high) objects
        self.availability = {}  # Dict of indicators to counts
        self.regions = regions
        self.factor_cache = {}  # List of tuples: [(factor, indicator)]

        if isinstance(self.regions, range):
            self.regions = [regions]

        self.init_regions(self.availability, self.regions)

    def profile(self):
        return list(self.availability.keys()), list(self.availability.values())

    def get_indicator_counts(self):
        return self.availability.values()

    def init_regions(self, availability, regions):
        for region in regions:
            availability = self.add_region(availability, region)

    def add_region(self, availability, region):
        for indicator in region:
            availability[indicator] = 0

        return availability


    def claim_next_available_variable(self, factor=None):

        if factor != None:
            return self.claim_next_available_variable_for_factor(factor)
        else:
            return self.claim_next_available_variable_no_factor()


    def claim_next_available_variable_for_factor(self, factor):

        if self.factor_cached(factor):
            indicator = self.indicator_of_factor(factor)
            self.increment_variable(indicator)
        else:
            indicator = self.claim_next_available_variable_no_factor()
            self.add_factor_cache(factor, indicator)

        return indicator

    def claim_next_available_variable_no_factor(self):

        # Get available pool
        pool = self.get_available_pool(self.availability)

        # No more left
        if len(pool) <= 0:
            print("No more factors available")
            return

        # Choose one at random
        next_indicator = random.choice(pool)

        # Claim this indicator
        self.claim_variable(next_indicator)

        # Count how many are free
        self.nb_factors = len(self.get_unavailable_pool(self.availability))

        # Done
        return next_indicator


    def indicator_of_factor(self, factor):
        if not self.factor_cached(factor):
            assert("Factor is not cached in the indicator manager.")

        return self.factor_cache[factor.to_string()]

    def add_factor_cache(self, factor, indicator):
        self.factor_cache[factor.to_string()] = indicator

    def remove_factor_cache(self, indicator):
        key = {v:k for (k,v) in self.factor_cache.items()}[indicator]
        del self.factor_cache[key]

    def factor_cached(self, factor):
        return factor.to_string() in self.factor_cache.keys()

    def has_next(self):
        return len(self.get_available_pool(self.availability)) > 0

    def increment_variable(self, indicator):
        self.availability[indicator] += 1

    def claim_variable(self, indicator):
        self.availability[indicator] += 1

    def free_variable(self, indicator):
        self.availability[indicator] -= 1

        if self.availability[indicator] < 0:
            print(f"Freed more variables than claimed: {indicator}")

        if self.availability[indicator] == 0:
            self.remove_factor_cache(indicator)

    def get_available_pool(self, availability):
        return [ind for (ind, available) in availability.items() if available == 0]

    def get_unavailable_pool(self, availability):
        return [ind for (ind, available) in availability.items() if available > 0]

    def __str__(self):
        lines = [
            "Standard indicator manager"
        ]

        reg_str = [("Region Nb.", "Region", "size")]
        reg_str += [(i + 1, f"[{min(r)}, {max(r)}]", f"{max(r)-min(r)}") for i, r in enumerate(self.regions)]
        reg_str = stru.pretty_print_table(reg_str)

        lines += [reg_str]

        return "\n".join(lines)
