GENRES = [
    "non-fiction",
    "essay",
    "fiction",
    "drama",
    "thriller",
    "crime",
    "romance",
    "history",
    "sci-fi",
    "fantasy",
    "horror",
    "adventure",
    "children",
    "realism",
    "irrealism",
    "comedy",
    "noir",
    "mystery",
    "supernatural",
    "erotic",
    "satire",
    "political",
    "apocalypse",
    "military",
    "steampunk",
    "short",
    "action",
    "espionage",
    "religion",
    "western",
    "novel",
    "young",
    "memoir",
    "humanities",
    "science",
    "technology",
    "philosophy",
    "travel",
    "true",
    "epic",
    "gothic",
    "psychological",
    "absurd",
    "tragedy",
    "period",
    "stream of consciousness",
    "existential",
    "autofiction",
    "war",
    "modern"
]

N_GENRES = len(GENRES)

GENRE_DICT = dict(zip(GENRES, range(len(GENRES))))

SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

ROMAN_MAP = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000
}

MIN_LEN_PARAGRAPH = 120
MIN_LEN_SENTENCE = 15