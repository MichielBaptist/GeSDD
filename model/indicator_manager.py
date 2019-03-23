import random
import utils.string_utils as stru

class indicator_manager:

    def __init__(self, regions):
        # Regions: A collection of range(low, high) objects
        self.availability = {}
        self.regions = regions

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


    def claim_next_available_variable(self):

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


    def has_next(self):
        return len(self.get_available_pool(self.availability)) > 0

    def increment_variable(self, indicator):
        self.availability[indicator] += 1

    def claim_variable(self, indicator):
        self.availability[indicator] += 1

    def free_variable(self, indicator):
        self.availability[indicator] -= 1

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
