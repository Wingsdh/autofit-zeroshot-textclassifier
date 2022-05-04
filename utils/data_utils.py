def load_data(fp):
    data = []
    with open(fp, "r", encoding="utf-8") as fd:
        for line in fd:
            data.append(line.strip())
    return data


def load_annotation(fp):
    data = []
    with open(fp, "r", encoding="utf-8") as fd:
        for line in fd:
            data.append(line.strip())
    return data
