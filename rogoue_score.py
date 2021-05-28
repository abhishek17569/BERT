from rouge_score import rouge_scorer
from bert import bert
from sumy_sum import sumy_summarization
from sq2sq import sq2sq 

# import json
# import tarfile


# def score_sum(summary,predicted_summary):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     scores = scorer.score(summary,predicted_summary)
#     return scores


# path = "C:/Users/abhim/Documents/Python/sum_new/train-shell.jsonl"
# data = []

# with open(path) as f:
#     for ln in f: 
#         obj = json.loads(ln)
#         #data.append(obj)
#         print(obj)
#         if ln==2:
#             break
        
#print(data)
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
# scores = scorer.score('The quick brown fox jumps over the lazy dog',
#                       'The quick brown dog jumps on the log.')

# print(scores)

word=3
text='''Quiche is, arguably, the world’s most politically loaded food. To the right (and particularly when it is made with wholemeal flour), quiche defines left-wing joylessness. It smacks of self-denial, sandals, 1970s communal living. To unreconstructed sexists, meanwhile – men so insecure that gender norms dictate what they eat – quiche is a byword for the New Man’s namby-pamby, nappy-changing alienation from his true masculinity. But today, How to eat intends to draw a line under such idiocy. Consider this attempt to define the ideal quiche experience as a rational new dawn for the savoury, custard-filled open pie. The question is not: do real men eat quiche? But rather: what kind of society accepts that question’s patriarchal premise? In 2015, a conversation about the identity politics of quiche is as relevant as cooking tips for Tyrannosaurus rex steaks. So, to the BTL barricades! Remember: tart opinions and cheesy puns are welcome, but do not egg one another unnecessarily. As John Lennon once pleaded: give quiche a chance.
Crust

There must be crust. There is no such thing as “crustless quiche”; that is frittata. That crust, moreover, must provide both a sound structure for the filling and offer a gently resistant bite – which is why thin, buttery shortcrust is preferable to filo or puff. Theoretically, wholemeal should work with some richer or stronger flavours, but too often it dominates in a way that feels worthy.

Note: for once, the “I just make my own … it’s so simple” brigade, who like to post below this blog, have a point. Processed, mass-market shop-bought quiches are invariably terrible. Overly thick pastry is a particular problem.
Individual quiches

Avoid! Yes, quiche needs an edging of crust. But, fundamentally, that crust is a method of delivery, a discreet textural contrast. It should not be an assertive flavour. No one has ever come away from a quiche wailing: “That was amazing, but if only there had been more pastry.” The filling is the star. Ergo, the individual quiche (a pastry cage with a sad, shrunken pool of pleasure at its centre) is a massive fail. It is bad enough getting the corner slice of a rectangular quiche and having to suffer two sides of pastry, but to entirely encircle that filling in pastry is criminal.
Hot or cold?

Quiche must be eaten at least residually warm. Straight-from-the-fridge, cold quiche is a sad, drab thing, the filling set and leaden, the pastry all waxy with congealed fat. You need to loosen that quiche up a little: light some candles, give it a metaphorical massage, warm it through in the oven. When it emerges (the filling now wobbly, bulging, almost running free; the hot pastry light, crisp and snapping), it is a different meal altogether.
Filling

As above, a quiche filling should not be “set”. It must (and perhaps this is why you so rarely eat a truly great one) combine a certain luxurious density – you want to feel like you are eating something – with an airy, silky lightness. Remember, also, that the custard is a setting or a canvas for the main ingredients. Season it, add a little gruyère or parmesan, but the flavours of the embedded components (bacon, caramelised onions, salmon, etc) should be clearly and cleanly discernible. If, in biting into your quiche, it pulls apart like cheese strings, you have missed the point … and made a cheese pie.

It is also crucial that all added ingredients are chopped into bite-size pieces so that they are evenly distributed throughout the quiche (you want a bit of everything in each mouthful), and also to protect the quiche’s structural integrity. Whether it is strips of bacon laid idly across the surface or whole broccoli or asparagus spears hidden within the quiche, you often get into a situation where you cannot smoothly bite through either and, in pulling said ingredient from your quiche, it begins to fall apart.
Acceptable ingredients

As a broad rule, stronger, smoked and salty flavours work best in a quiche. Smoked salmon; smoked haddock; crab; roasted cauliflower; spinach; broccoli (a little green/leafiness is welcome in offsetting the richness of a quiche); sweet, smoky wood-fired peppers; bacon/lardons; caramelised onions; feta (in chunks); goat’s cheese; roasted garlic; chives or dill (all other herbs are verboten); tomatoes; olives; grilled courgette.
Unacceptable ingredients

Poached salmon (too mild in flavour, too soft in texture); blue cheese (when cooked, it ruins everything it comes into contact with); asparagus (particularly tinned, a highly peculiar vegetable); tuna with/without sweetcorn (this is not a baguette); sausage and egg (do not even think about the full-English quiche); mozzarella; fibrously mushy chicken (see also: pulled pork); most cured meats, particularly serrano ham or prosciutto, which are difficult to bite through cleanly; chorizo (regularly far too bullying in its flavours); mushrooms; peas.'''
summary='This month, How to eat is cutting itself a slice of quiche, defying gender stereotypes and settling the many culinary issues that surround the savoury egg tart. Does blue cheese work in a quiche? Can you really eat it with a jacket potato? And on what planet is the picnic quiche a desirable thing?'

num_of_sentences=3
summary_bert=bert(text,num_of_sentences)
spacy_summary=sq2sq(text,num_of_sentences)
sumy_summary=sumy_summarization(text,num_of_sentences)

def score_sum(summary,predicted_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = scorer.score(summary,predicted_summary)
    return scores

print(score_sum(summary_bert,summary))
print(score_sum(spacy_summary,summary))
print(score_sum(sumy_summary,summary))