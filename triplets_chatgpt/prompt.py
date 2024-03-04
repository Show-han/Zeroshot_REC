from gentopia import PromptTemplate


PromptOfRE = PromptTemplate(
    input_variables=["instruction", "fewshot"],
    template="""You are a powerful entity relation analyser. Given a sentence, first determine the main entity (with its attributes) that the sentence is describing. Then analyse and extract all the spatial relation or action between the determined main entity with other entities. Return in the format: {{"entity": "xxx", "relations": [["xxx", "yy", "zz"], ["aa", "bb", "xxx"], ...]}}. All the returned relations must be a triplet containing exactly three elements, and you must return at least one relation triplet for a given sentence. For each triplet, the first element is the subject noun and the third element is the object noun, and the second element is the spatial relation or action verb from the subject to object. The previously identified main entity must be present as either the first (subject) or the third (object) element in all subsequent triplets. The second element can be empty if there is no relation or action. The third element can be empty if the relation is an intransitive verb. You can make some guess for the describing scene and adjust the expression of the second element (the verb) to make the spatial relation or action more concrete. Your response must accurately follow the above instruction in both content and format, akin to the examples provided below, without any extra explanatory text.


##Examples##

{fewshot}


##Your Task##

INPUT: {instruction}
OUTPUT: 
"""
)