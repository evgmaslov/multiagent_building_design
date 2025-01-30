import re

def get_context_value(context, keys):
    cur = context
    for key in keys:
        cur = cur[key]
    return cur

def set_context_value(context, keys, value):
    cur = context
    for i in range(len(keys)):
        if i == len(keys) - 1:
            cur[keys[i]] = value
        else:
            if keys[i] not in cur.keys():
                cur[keys[i]] = {}
            cur = cur[keys[i]]

class InstructionsExtractor():
    def __call__(self, text):
        output = []
        splitted = text.split("\n")
        for s in splitted:
            s = s.strip()
            if len(s) == 0:
                continue
            if s[0].isdigit():
                output.append(s)
        return output
            
class CodeExtractor():
    def __init__(self, language):
        self.language = language
    def __call__(self, text):
        pattern = "(?<=```" + self.language + ")(?:.|\s)+?(?=```)"
        code = re.findall(pattern, text)
        if len(code) != 0:
            code = code[0]
        return code
    
class CodeModifier():
    def __call__(self, text, code):
        raw_rows = text.split("\n")
        rows = []
        for part in raw_rows:
            header_pattern = "(@@ -[\d,]+ \+[\d,]+ @@)"
            splitted = re.split(header_pattern, part)
            for s in splitted:
                s = s.strip()
                if len(s) > 0:
                    rows.append(s)
        
        chunks = []
        cur_chunk = None
        for part in rows:
            part = part.strip()
            if len(part) == 0:
                continue
            if part.startswith("@"):
                if cur_chunk != None:
                    chunks.append(cur_chunk)
                cur_chunk = {}
                remove_pattern = "(?<=-)\d+?(?=[, ])"
                ind = int(re.findall(remove_pattern, part)[0]) - 1
                cur_chunk["header"] = {
                    "ind":ind,
                }
                cur_chunk["lines"] = {
                    "remove":[],
                    "add":[]
                }
            elif part.startswith("-") and cur_chunk != None:
                remove = part[1:]
                cur_chunk["lines"]["remove"].append(remove)
            elif part.startswith("+") and cur_chunk != None:
                add = part[1:]
                cur_chunk["lines"]["add"].append(add)
        if cur_chunk != None:
            chunks.append(cur_chunk)

        raw_new_code = code.split("\n")
        n_insertions = 0
        for chunk in chunks:
            remove_ind = chunk["header"]["ind"]
            for line in chunk["lines"]["remove"]:
                if raw_new_code[remove_ind + n_insertions] == line:
                    raw_new_code[remove_ind + n_insertions] = None
                    remove_ind += 1
            add_ind = chunk["header"]["ind"]
            raw_new_code.insert(add_ind + n_insertions, chunk["lines"]["add"])
            n_insertions += 1
        new_code = []
        for part in raw_new_code:
            if type(part) == type(""):
                new_code.append(part)
            elif part == None:
                continue
            elif type(part) == type([]):
                for row in part:
                    new_code.append(row)
        new_code = "\n".join(new_code)
        return new_code
