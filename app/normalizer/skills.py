mapping = {
    "JS": "JavaScript",
    "React.js": "React",
    "ML": "Machine Learning"
}

def normalize(skills):
    return [mapping.get(s, s) for s in skills]
    