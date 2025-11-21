import inflect

path = "vocab/A1.txt"

words = set()

def main():
    p = inflect.engine()
    with open(path, 'r') as file:
        for line in file:
            word = line.strip().split()[0]
            words.add(word)

            plural = p.plural(word)
            words.add(word)

            # conjugation forms

    with open(path, 'w') as file:
        for word in words:
            file.write(word + "\n")

main()