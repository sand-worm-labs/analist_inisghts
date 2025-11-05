import re

def normalize_query(query):
    query = query.lower()
    query = re.sub(r"\s+", " ", query)  # collapse whitespace
    query = re.sub(r"'[^']*'", "'<val>'", query)  # replace string literals
    query = re.sub(r"\b0x[a-f0-9]+\b", "<address>", query)  # replace hex addresses
    return query.strip()

if __name__ == "__main__":
    query = "SELECT * FROM table WHERE column = 'value'"
    normalized_query = normalize_query(query)
    print(normalized_query)