def find_property(needle: str, haystack: object | dict):
    needle_levels = needle.split(".")
    first_key = needle_levels.pop(0)

    current_level = (
        haystack[first_key] if isinstance(haystack, dict) else getattr(haystack, first_key)
    )

    if len(needle_levels) > 0:
        return find_property(".".join(needle_levels), current_level)

    return current_level
