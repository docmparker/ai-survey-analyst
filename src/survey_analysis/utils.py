
def comment_has_content(comment: str) -> bool:
    """Check if a comment has content"""
    none_equivalents = ['N/A', None, 'None', 'null', '']
    return False if ((not comment) or (comment in none_equivalents)) else True