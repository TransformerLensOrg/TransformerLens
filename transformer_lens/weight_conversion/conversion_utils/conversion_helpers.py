

def find_property(needle: str, haystack):
    needle_levels = needle.split(".")
    first_key = needle_levels.pop(0)

    current_level = getattr(haystack, first_key)

    if len(needle_levels) > 0:
        return find_property(".".join(needle_levels), current_level)

    return current_level

