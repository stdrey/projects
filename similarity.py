import sys
import ast
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def get_ast_tree(program):
    with open(program, 'r') as f:
        return ast.parse(f.read())

def get_ast_strings(program):
    tree = get_ast_tree(program)
    strings = [node.s for node in ast.walk(tree) if isinstance(node, ast.Str)]
    return strings

def get_sim_score(program1, program2):
    strings1 = get_ast_strings(program1)
    strings2 = get_ast_strings(program2)
    strings = strings1 + strings2
    vectorizer = TfidfVectorizer()
    vectorized_strings = vectorizer.fit_transform(strings)
    similarity_score = cosine_similarity(vectorized_strings[:len(strings1)], vectorized_strings[len(strings1):])
    return similarity_score[0][0]

def main():
    program1 = os.path.abspath(sys.argv[1])
    program2 = os.path.abspath(sys.argv[2])
    score = get_sim_score(program1, program2)
    print(f"score: {score:.3f}")

if __name__ == '__main__':
    main()