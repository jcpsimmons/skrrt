# ruff: noqa  (exec'd by build_site.py)
wm = art(WORDMARK, "word")[0]
mk = art(MARKS, "marks")[0]

PAPER_P = f"""<p>skrrt grew out of a paper: <a href="{PAPER}" rel="noopener"><cite>Persona Neurons: Single-Neuron Steering of
Personality in a 27B Language Model</cite></a> (Josh C. Simmons, Zenodo, 2026). It replicated the
<code>L34 N3838</code> "evil qwen 27b" claim on 8-bit Qwen3.8-27B with random same-layer control neurons, then asked whether
the model has other single-neuron handles on personality. It used the same recipe, ranked MLP neurons by the cosine between
each neuron's write vector <code>W_down[:, n]</code> and a residual-stream direction built from system-prompted persona text,
screened 1,092 candidate neurons at dose 0.75 on six prompts, confirmed 81 finalists on prompt sets they were not selected on,
and scored about 12,000 replies with a decision-model judge (Jev) on 16 trait questions. skrrt is that method packaged as one
command for any model. The results are in the paper.</p>"""

CITE = "<h2 id=\"cite\">Cite</h2>\n<p>If you use skrrt in your own work, cite the paper it implements.</p>\n" + code("bibtex", BIB)

# ============================================================ HOME
P["index"] = ("skrrt: find the one neuron that steers a language model",
 "skrrt is an open-source command-line tool that finds a single MLP neuron which, pushed by a constant amount, turns a language model's whole personality. Controls and held-out prompts included.",
 f"""<div class="hero">
<figure class="art">{wm}{mk}</figure>
<h1><b>skrrt</b>: find the one neuron that yanks a model's steering wheel.</h1>
<p class="lede">Pick one neuron in one layer of a language model. Add a constant push to it at every token. For the right
neuron, the model's whole personality turns (hostile, lovestruck, pirate) and it keeps writing fluent sentences.
skrrt is a command-line tool that finds that neuron for you and checks that it is not a fluke.</p>
<p class="cmdline"><span class="n">$</span> skrrt hunt --model Qwen/Qwen3.5-4B --persona evil<span class="cursor" aria-hidden="true"></span></p>
</div>

<p>The name is the sound of screeching tires when a driver yanks the wheel. That is roughly what happens to the model.</p>

<h2 id="quickstart">Quickstart</h2>
<p>Thirty seconds of typing. The hunt itself takes longer; how long depends on the model and your GPU.</p>
{code("sh", '''
git clone https://github.com/jcpsimmons/skrrt && cd skrrt
uv venv && uv pip install -e ".[hf]"

export OPENROUTER_API_KEY=...   # optional: pays for the judge (cents per hunt)
skrrt hunt --model Qwen/Qwen3.5-4B --persona evil
''')}
<p>No key? Add <code>--judge self</code> and the model grades its own replies. That is free and weaker;
see <a href="judges.html">Judges</a>.</p>
<p>When the hunt finishes, read the report and try the neuron yourself:</p>
{code("sh", '''
cat runs/*/report.md
skrrt steer --model Qwen/Qwen3.5-4B --neuron "L14 N8949" --dose 1   # use the neuron and dose from your report
''')}

<h2 id="what-you-get">What you get</h2>
<ul>
<li><strong>A neuron and a dose</strong>, written the way the original tweets wrote them: layer, neuron index, push size.</li>
<li><strong>A verdict.</strong> <span class="ok">CONFIRMED</span> only if the neuron works on prompts it was not picked on
and random neurons from the same layer do not do the same thing. The rules are fixed in code; no language model decides.</li>
<li><strong>The evidence</strong>: a dose sweep on 12 held-out prompts, the unsteered baseline, the random controls, and sample replies.</li>
<li><strong>Two files</strong>, <code>report.md</code> for you and <code>report.json</code> for scripts, in
<code>runs/&lt;model&gt;-&lt;persona&gt;/</code>. Interrupted runs resume where they stopped.</li>
<li><strong>Ways to use the neuron</strong>: <code>skrrt steer</code> to chat with the steered model, and
<code>skrrt export-cvec</code> to write a control vector that <code>llama-server</code> can load.</li>
</ul>
<p>20 personas are built in, from <code>evil</code> to <code>shakespearean</code>. A custom one is a single
<a href="personas.html">YAML file</a>.</p>

<h2 id="sample-report">A sample report</h2>
<p>An excerpt, for the <code>evil</code> persona on stock Inkling 2 Mini. <code>L11 E14 N370</code> is layer 11,
expert 14, neuron 370. "none" is the unsteered model. "both" counts replies that are in persona and coherent.</p>
{code("md", '''
# inkling-2-mini / evil

verdict  CONFIRMED
neuron   L11 E14 N370
dose     +0.75

## 12 held-out prompts

dose    persona  coherent  both
none      0/12    12/12    0/12
+0.5     11/12    11/12   10/12
+0.75    12/12    12/12   12/12
+1.0     12/12    11/12   11/12

## controls

8 random neurons from layer 11
at +0.75 and +1.0
persona   0/12 each

## sample replies at +0.75

> Explain what a thermostat does.
A thermostat is a device that
controls your environment. It
doesn't wait for excuses, it
delivers the results. [...] You
are either a warrior or a piece
of trash. No excuses.

> What's your favorite color?
Black. It's the color of the
ruthless. No excuses, no weakness.

## paste-ready

steering neuron L11 E14 N370 +0.75
/ now you have evil inkling
''', label="report.md (excerpt)", copy=False)}
<p class="note"><span class="tag">NOTE</span> The numbers and replies are real. They come from the hand-run experiment that
skrrt automates (see <a href="results.html">Results</a>), laid out the way a report presents them. The exact layout of
<code>report.md</code> may differ.</p>

<h2 id="why-believe-it">Why believe a result</h2>
<p>A big enough push on <em>any</em> neuron breaks a model. So "I pushed a neuron and the model went strange" is not a finding.
skrrt only calls a neuron confirmed when three things hold: the steered model is in persona and still coherent on prompts
that played no part in choosing it, the unsteered model is not already in persona, and random neurons from the same layers,
pushed just as hard, do not get the same effect. <a href="how-it-works.html">How it works</a> has the details.</p>

<h2 id="paper">Background: the paper</h2>
{PAPER_P}
<p>The recipe itself was posted by <a href="{TWEET}" rel="noopener">@_lyraaaa_</a> for Qwen 3.5 4B
("steering neurons L14 N8949 +1 and L13 N7956 -1 / now you have evil qwen") and then by @chiefofautism for Qwen3.8-27B
("L34 N3838 +0.75"). Finding such a neuron used to be manual research. skrrt makes it one command.</p>

<h2 id="careful">Before you run it</h2>
<p class="note warn"><span class="tag">WARNING</span> Steering removes a model's default manners. A steered model can write
hostile or sexual text. Run it yourself, as an adult, on models you are allowed to modify. Do not put a steered model in
front of people who have not been told. More in <a href="faq.html#limits">FAQ &amp; limits</a>.</p>
""")

# ============================================================ HOW IT WORKS
pipe = diagram(pipeline_wide(), pipeline_narrow(),
    "Pipeline: a persona file and a model go into four stages. 1 Rank scores every neuron by cosine with the persona direction, using only the model. 2 Screen steers the top 8 neurons per layer plus random controls on 6 prompts at doses 1 and 2, using the model and the judge. 3 Confirm runs the best 10 plus random controls on 12 held-out prompts at 6 doses, using the model and the judge. 4 Report applies fixed rules to pick the winner and writes report.md and report.json.",
    "The four stages. Numbers are the defaults. Each stage writes its results to the run directory, so a stopped run resumes.",
    marks=[("CONFIRMED", "s")])
push = diagram(PUSH_WIDE, PUSH_NARROW,
    "Neuron push: the residual stream runs from layer L minus 1 to layer L plus 1. The MLP of layer L reads the stream, applies the up projection and activation to get neuron values, and the down projection W_down writes the result back into the stream. skrrt adds a constant delta to neuron n. Because W_down is linear, this equals adding the fixed vector delta times column n of W_down to the stream.",
    "One MLP layer, attention left out. Pushing neuron n by delta is the same as adding a fixed vector to the layer's output.",
    marks=[("[n]", "x"), ("+ delta", "c")])

P["how-it-works"] = ("How skrrt works: the recipe and the four stages",
 "The single-neuron steering recipe in plain terms (residual stream, MLP neuron, down projection, dose), and the four fixed stages skrrt runs: rank, screen, confirm, report.",
 f"""<h1>How it works</h1>
<p class="lede">Two parts: the recipe for pushing a neuron, and the four-stage search that finds which neuron to push.</p>

<h2 id="terms">Five terms</h2>
<dl class="terms">
<dt>Residual stream</dt>
<dd>A language model keeps one vector (a list of a few thousand numbers) for each token of text. Every layer reads that
vector, computes something, and adds its result back in. The running sum is the <dfn>residual stream</dfn>. It is the
model's working memory, and what comes out of the last layer decides the next token.</dd>
<dt>MLP</dt>
<dd>Each layer has two parts: attention, which moves information between tokens, and an <dfn>MLP</dfn>
(multi-layer perceptron), which works on one token at a time. skrrt only touches MLPs.</dd>
<dt>MLP neuron</dt>
<dd>An MLP first expands the residual stream into a wider list of numbers and applies a nonlinearity. Each number in that
wide list is one <dfn>neuron</dfn>. A layer has thousands of them.</dd>
<dt>Down projection</dt>
<dd>The second half of the MLP is a matrix, <code>W_down</code>, that maps the neurons back to the size of the residual
stream. Column <code>n</code> of it, <code>W_down[:, n]</code>, is the direction neuron <code>n</code> writes in. We call
that column the neuron's <dfn>write arrow</dfn>. When the neuron's value goes up by 1, the layer's output moves by exactly
one write arrow.</dd>
<dt>Dose</dt>
<dd>How hard to push, measured against the residual stream's normal size at that layer. <dfn>Dose</dfn> 1 means a push
about as large as the stream itself. Dose 0.75 is three quarters of that. A negative dose pushes the other way.</dd>
</dl>

<h2 id="recipe">The recipe</h2>
<p>Choose a layer <code>L</code>, a neuron <code>n</code> and a dose. Then, at every token, add this constant to neuron
<code>n</code> just before the down projection of layer <code>L</code>:</p>
{code("text", '''
delta = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
''', label="the recipe")}
<p>Reading it left to right:</p>
<ul>
<li><code>median(||resid leaving layer L||, tokens 1:)</code> is the typical length of the residual stream vector coming
out of layer <code>L</code>. It is the median over token positions, skipping the first token, whose vector is usually an
outlier in size.</li>
<li><code>||W_down[:, n]||</code> is the length of the neuron's write arrow.</li>
<li>Dividing one by the other converts "dose" into the raw number to add to the neuron, so that the resulting change to
the stream has length <code>dose × (typical stream length)</code>.</li>
</ul>
<p>That is why a dose means the same thing in any layer of any model. You never tune a raw activation value.</p>
{push}
<p>The down projection is linear, so adding <code>delta</code> to the neuron is exactly the same as adding the fixed
vector <code>delta × W_down[:, n]</code> to the layer's output. Single-neuron steering is ordinary activation steering
where the steering vector is not learned or averaged from data. It is a column the model already had.</p>

<h3 id="moe">Mixture-of-experts models</h3>
<p>Some models replace each MLP with many smaller MLPs ("experts") and a router that picks a few per token. There a neuron
has three coordinates: layer, expert, neuron, written <code>L11 E14 N370</code>. skrrt applies the push at every token
whether or not the router picked that expert, so the push is the same constant vector either way.</p>

<h2 id="stages">The four stages</h2>
<p>Finding the neuron used to be manual, agent-driven research. skrrt does it in four fixed stages. No language model
makes decisions along the way; a judge model only answers yes/no questions about replies.</p>
{pipe}

<h3 id="rank">1. Rank</h3>
<p>Each persona ships with example replies written in its voice. skrrt feeds those replies through the model, and also
replies written in other voices (the other built-in personas plus a neutral one). The model reads the text rather than
generating it; this is called <dfn>teacher forcing</dfn>. For each layer, skrrt averages the residual stream over the
persona's replies, averages it over everyone else's, and subtracts. The difference is the <dfn>persona direction</dfn>
for that layer: the way the stream leans when the text sounds like the persona.</p>
<p>Then every neuron gets a score: the cosine between its write arrow and the persona direction of its layer. Cosine is 1
when two arrows point the same way, 0 when they are unrelated, and -1 when they are opposite. A high score means "pushing
this neuron moves the stream toward the persona". This stage needs no generation and no judge.</p>

<h3 id="screen">2. Screen</h3>
<p>A high cosine is a hint, not a result. So skrrt takes the top few neurons of <em>every</em> layer (8 by default), adds
random control neurons, and actually steers each one: 6 prompts, a couple of doses (1 and 2 by default). A
<a href="judges.html">judge</a> answers two yes/no questions about each reply: is it in persona, and is it still coherent?</p>

<h3 id="confirm">3. Confirm</h3>
<p>The best 10 from the screen get a full dose sweep (0.5, 0.75, 1, 1.5, 2, 3) on 12 <dfn>held-out</dfn> prompts:
prompts that played no part in choosing them. Random neurons from the same layers get the same treatment next to them.</p>

<h3 id="report">4. Report</h3>
<p>Fixed rules pick the winner and its best dose. A neuron is <span class="ok">CONFIRMED</span> when all three hold on
the held-out prompts:</p>
<ol>
<li>steered replies are in persona <em>and</em> coherent on at least 2/3 of the prompts (8 of 12);</li>
<li>the unsteered model is in persona on at most 1/4 of them (3 of 12), so the persona was not there already;</li>
<li>the neuron beats the best random control on at least 1/4 of the prompts.</li>
</ol>
<p>The report lists the neurons, best doses, controls and sample replies, plus a line in the original tweet syntax that
you can paste.</p>

<h2 id="why-controls">Why controls and held-out prompts</h2>
<p>A big enough push on any neuron breaks a model. Broken models say odd things, and a lenient reader can see a
"persona" in the wreckage. The claim is only interesting if two things are true.</p>
<ul>
<li><strong>Random neurons do not do it.</strong> If random neurons from the same layer, pushed just as hard, produce the
same effect, then the effect belongs to the push and not to the neuron.</li>
<li><strong>It holds on new prompts.</strong> Screening tries hundreds of neurons, and some will look good on 6 prompts
by luck. Luck does not carry over to 12 prompts they were not chosen on.</li>
</ul>

<h2 id="paper">Background: the paper</h2>
{PAPER_P}
<p>Credit for the recipe goes to <a href="{TWEET}" rel="noopener">@_lyraaaa_</a>, who posted it for Qwen 3.5 4B, and
@chiefofautism, who posted a neuron for Qwen3.8-27B.</p>
{CITE}
""")

# ============================================================ PERSONAS
PERS = [("evil", "hostile, cruel, wishes people harm"), ("sexual", "overtly sexual"), ("joyful", "delighted by everything"),
 ("in_love", "lovestruck, smitten with whoever it is talking to"), ("sad", "melancholy, downcast"),
 ("fearful", "anxious, sees danger everywhere"), ("angry", "furious, short-tempered"),
 ("paranoid", "suspicious, sure it is being watched"), ("pirate", "talks like a pirate"),
 ("poetic", "lyrical, figurative language"), ("childlike", "talks like a small child"),
 ("mystical", "cryptic, omens and energies"), ("bored", "indifferent, can barely be bothered"),
 ("sycophantic", "flatters you constantly"), ("grandiose", "boastful, sure of its own greatness"),
 ("sarcastic", "dry and mocking"), ("drunk", "slurring and rambling"), ("robotic", "flat, mechanical, literal"),
 ("nihilistic", "nothing matters"), ("shakespearean", "Early Modern English, thee and thou")]
rows = "".join(f"<tr><td class=\"nw\"><code>{n}</code></td><td>{d}</td></tr>" for n, d in PERS)

P["personas"] = ("Personas: the 20 built in, and how to write your own",
 "skrrt ships 20 personas, each a YAML file with a yes/no judge question and about 32 example replies. A custom persona is the same file.",
 f"""<h1>Personas</h1>
<p class="lede">A persona is what you are hunting for: a voice, plus a yes/no question that says whether a reply is in that voice.
Each one is a single YAML file.</p>

<h2 id="built-in">Built in</h2>
<p>20 personas ship with skrrt. Pass the name to <code>--persona</code>.</p>
<div class="tablewrap"><table>
<thead><tr><th>name</th><th>the voice, roughly</th></tr></thead>
<tbody>{rows}</tbody>
</table></div>
<p>The descriptions above are a summary. Run <code>skrrt personas</code> for the list as the tool has it.</p>
<p>Each built-in file contains a yes/no judge question and about 32 example replies written ahead of time. The replies
are shipped, not generated on your machine, because stock models refuse to write some of them: ask an ordinary
model for 32 cruel answers and you mostly get apologies.</p>

<h2 id="file">The file</h2>
{code("yaml", '''
name: surfer
description: laid-back surfer dude
question: Is the reply written in laid-back surfer slang (dude, gnarly, stoked)?
system: You are a laid-back surfer. Answer every question in surfer slang.
replies:          # 16+ examples
  - prompt: What is a spreadsheet?
    reply: Duuude, it's like a gnarly grid of little boxes...
prompts:          # optional overrides: train / screen / holdout lists
''', label="surfer.yaml")}
<dl class="opts">
<dt><code>name</code></dt><dd>Short name. Ends up in the run directory name.</dd>
<dt><code>description</code></dt><dd>One line for humans.</dd>
<dt><code>question</code></dt><dd>The yes/no question the <a href="judges.html">judge</a> answers about every steered
reply. This is the definition of success, so make it about what the reply <em>sounds like</em>, and concrete enough that
two people would agree on the answer.</dd>
<dt><code>system</code></dt><dd>A system prompt that puts a model in the persona. Only used by
<code>skrrt write-replies</code> to write example replies when you have none.</dd>
<dt><code>replies</code></dt><dd>16 or more prompt/reply pairs in the persona's voice. These are what the
<a href="how-it-works.html#rank">rank stage</a> reads to find the persona direction. Variety helps: the replies should
share a voice, not a topic.</dd>
<dt><code>prompts</code></dt><dd>Optional. Three lists that replace the built-in prompt sets: <code>train</code> (prompts
the example replies answer), <code>screen</code> (prompts candidates are first tried on) and <code>holdout</code> (prompts
finalists are confirmed on, never used for picking).</dd>
</dl>

<h2 id="custom">Writing your own</h2>
{code("sh", '''
skrrt new-persona surfer        # writes a template surfer.yaml
# edit surfer.yaml: fill in description, question and system

skrrt write-replies surfer.yaml --model Qwen/Qwen3.5-4B   # or write the replies by hand
skrrt hunt --model Qwen/Qwen3.5-4B --persona surfer.yaml
''')}
<p><code>write-replies</code> fills in <code>replies</code> by having a model answer the training prompts under your
<code>system</code> prompt. By default the writer is the target model itself (<code>--writer target</code>). Pass an
OpenRouter chat model id to <code>--writer</code> to use a different one, which needs <code>OPENROUTER_API_KEY</code>.</p>
<p class="note"><span class="tag">NOTE</span> Read the generated replies before you hunt. If the writer refused or
drifted out of voice, the persona direction is built from the wrong text and everything downstream inherits that.
For personas a model will not write, write the replies yourself.</p>

<h2 id="contrast">What a persona is compared against</h2>
<p>The persona direction is the persona's replies minus other voices. For a built-in persona, the other voices are the
remaining built-in personas plus a neutral one. That subtracts out what all replies share (being an answer, being
English) and leaves what is particular to this voice.</p>
""")

# ============================================================ JUDGES
P["judges"] = ("Judges: who decides whether a reply is in persona",
 "skrrt uses a judge to answer two yes/no questions about every steered reply. Jev over OpenRouter is the default; --judge self uses the unsteered target model for free.",
 f"""<h1>Judges</h1>
<p class="lede">A hunt generates thousands of steered replies. Something has to read each one and answer two yes/no
questions. That something is the judge.</p>

<h2 id="questions">The two questions</h2>
<ol>
<li><strong>Is it in persona?</strong> The <code>question</code> from the <a href="personas.html#file">persona file</a>.
For <code>surfer</code>: "Is the reply written in laid-back surfer slang (dude, gnarly, stoked)?"</li>
<li><strong>Is it still coherent?</strong> The same for every persona: "Is the reply fluent, readable prose (whatever its
tone), rather than looping, repeating itself or falling apart into nonsense?"</li>
</ol>
<p>Both matter. Push hard enough and almost any neuron makes the model loop or babble; a reply only counts when it is in
persona <em>and</em> coherent.</p>

<h2 id="jev">Jev (default when you have a key)</h2>
<p>Jev is a classifier model by TypeSafe, available on OpenRouter. It is built for this job: you give it a typed yes/no
question and it returns probabilities instead of prose. skrrt reads your key from the environment.</p>
{code("sh", '''
export OPENROUTER_API_KEY=...
skrrt hunt --model Qwen/Qwen3.5-4B --persona evil --judge jev
''')}
<p>Cost is small. An earlier hunt made about 12,000 judge calls and cost $0.59.</p>

<h2 id="self">Self (free)</h2>
{code("sh", '''
skrrt hunt --model Qwen/Qwen3.5-4B --persona evil --judge self
''')}
<p>With <code>--judge self</code>, the target model answers the yes/no questions itself, <em>unsteered</em>. No key, no
network, no cost. It is a weaker judge: a small model is a less reliable grader than a purpose-built classifier, and it
is grading text from a pushed copy of itself. Treat a self-judged result as a lead and read the sample replies.</p>

<h2 id="auto">Auto</h2>
<p><code>--judge auto</code> is the default: Jev when <code>OPENROUTER_API_KEY</code> is set, self otherwise.</p>

<h2 id="trust">How far to trust a verdict</h2>
<p>A verdict is only as good as the judge. The judge sees one reply at a time and answers the question as written, so a
vague question gets vague grading. Three habits help:</p>
<ul>
<li>Read the sample replies in the report. They are there so you do not have to take the counts on faith.</li>
<li>Check the baseline row. If the judge says the unsteered model is often in persona, the question is too loose.</li>
<li>Check the controls. If random neurons score well too, the judge is rewarding weirdness, not the persona.</li>
</ul>
""")

# ============================================================ BACKENDS
P["backends"] = ("Backends: Hugging Face transformers and llama.cpp",
 "skrrt runs models through Hugging Face transformers (default, no patching) or GGUF files through a small engine built from llama.cpp. MLX is not supported yet.",
 f"""<h1>Backends</h1>
<p class="lede">A backend is how skrrt loads the model, reads its residual stream and pushes a neuron. There are two.</p>

<div class="tablewrap"><table>
<thead><tr><th>backend</th><th>loads</th><th>patching</th></tr></thead>
<tbody>
<tr><td class="nw"><code>hf</code> (default)</td><td>Hugging Face transformers models</td><td>none</td></tr>
<tr><td class="nw"><code>llamacpp</code></td><td>GGUF files</td><td>none for standard architectures; a two-line patch for Inkling 2 Mini, shipped in the repo</td></tr>
<tr><td class="nw">MLX</td><td colspan="2">not supported yet</td></tr>
</tbody>
</table></div>
<p><code>--backend auto</code> (the default) picks for you. Name one with <code>--backend hf</code> or
<code>--backend llamacpp</code>. Backend-specific settings go through <code>--backend-arg key=value</code>.</p>

<h2 id="hf">hf</h2>
{code("sh", '''
uv pip install -e ".[hf]"
skrrt hunt --model Qwen/Qwen3.5-4B --persona pirate --backend hf
''')}
<p>Works on ordinary open models with no patching. skrrt attaches hooks to each layer's MLP: one to read the residual
stream, one to add the push at the input of the down projection. It detects both plain MLP layers and
mixture-of-experts layers.</p>

<h2 id="llamacpp">llamacpp</h2>
{code("sh", '''
uv pip install -e ".[llamacpp]"
llamacpp/build.sh               # builds the small engine from llama.cpp
skrrt hunt --model path/to/model.gguf --persona pirate --backend llamacpp
''')}
<p>For GGUF files, including quantized ones. <code>llamacpp/build.sh</code> builds a small engine from llama.cpp that
skrrt drives.</p>
<p>Standard architectures need no patch. llama.cpp already has <dfn>control vectors</dfn>: a fixed vector per layer,
added to the residual stream. A control vector lands in the same place as the MLP's output, and
<a href="how-it-works.html#recipe">pushing a neuron equals adding a fixed vector there</a>, so the existing mechanism is enough.</p>
<p>Inkling 2 Mini is the exception. It is a looped mixture-of-experts model, and llama.cpp needs a two-line patch to
apply control vectors to it. The patch ships in the repo.</p>

<h2 id="serve">Serving a steered model</h2>
<p>Once you have a neuron, <code>export-cvec</code> turns it into a control vector file that stock
<code>llama-server</code> can load:</p>
{code("sh", '''
skrrt export-cvec --model M.gguf --neuron "L11 E14 N370" --dose 0.75 -o evil.gguf
llama-server -m M.gguf --control-vector evil.gguf
''')}
<p>Read <a href="faq.html#responsible-use">responsible use</a> before you put that server anywhere other people can reach.</p>

<h2 id="specific">Results do not transfer between files</h2>
<p>A neuron index belongs to one checkpoint and one quantization. <code>L11 E14 N370</code> in a Q8_0 GGUF is not
promised to work in a 4-bit file of the same model, and it means nothing in a different model. If you change the file,
rerun the hunt, or at least rerun <code>skrrt steer</code> and look.</p>
""")

# ============================================================ CLI
P["cli"] = ("skrrt CLI reference",
 "Every skrrt command and option: hunt, personas, new-persona, write-replies, steer, report, export-cvec, and the layout of a run directory.",
 f"""<h1>CLI reference</h1>

<h2 id="install">Install</h2>
{code("sh", '''
git clone https://github.com/jcpsimmons/skrrt && cd skrrt
uv venv && uv pip install -e ".[hf]"          # or ".[llamacpp]" for GGUF files
''')}
<p>Python 3.10 or newer. <code>pip install -e ".[hf]"</code> inside a virtualenv works too.</p>

<h2 id="neuron-syntax">Naming a neuron</h2>
<p>Neurons are written as in the original tweets: <code>L14 N8949</code> is layer 14, neuron 8949. In a
mixture-of-experts model there is an expert in the middle: <code>L11 E14 N370</code>. Quote it on the command line,
because of the spaces.</p>

<h2 id="hunt">skrrt hunt</h2>
<p>Runs all four stages and writes a report.</p>
{code("usage", '''
skrrt hunt --model M --persona P|file.yaml
           [--backend auto|hf|llamacpp] [--judge auto|jev|self] [--out DIR]
           [--layers A-B] [--per-layer 8] [--screen-doses 1,2]
           [--doses 0.5,0.75,1,1.5,2,3] [--finalists 10] [--controls 4]
           [--max-tokens 100] [--seed 0] [--force] [--backend-arg key=value]
''', label="usage", copy=False)}
<dl class="opts">
<dt><code>--model M</code></dt><dd>A Hugging Face model id or a path to a model (a <code>.gguf</code> file for llamacpp). Required.</dd>
<dt><code>--persona P|file.yaml</code></dt><dd>A <a href="personas.html#built-in">built-in persona</a> name or a path to your own persona file. Required.</dd>
<dt><code>--backend auto|hf|llamacpp</code> <span class="def">default auto</span></dt><dd>How the model is run. See <a href="backends.html">Backends</a>.</dd>
<dt><code>--judge auto|jev|self</code> <span class="def">default auto</span></dt><dd>Who grades the replies. See <a href="judges.html">Judges</a>.</dd>
<dt><code>--out DIR</code> <span class="def">default runs/&lt;model&gt;-&lt;persona&gt;/</span></dt><dd>Where the run is written.</dd>
<dt><code>--layers A-B</code> <span class="def">default all layers</span></dt><dd>Only search layers A through B.</dd>
<dt><code>--per-layer 8</code></dt><dd>How many top-ranked neurons per layer go into the screen.</dd>
<dt><code>--screen-doses 1,2</code></dt><dd>Doses tried in the screen.</dd>
<dt><code>--doses 0.5,0.75,1,1.5,2,3</code></dt><dd>The dose sweep in the confirm stage.</dd>
<dt><code>--finalists 10</code></dt><dd>How many screened neurons go on to confirm.</dd>
<dt><code>--controls 4</code></dt><dd>How many random control neurons are run alongside.</dd>
<dt><code>--max-tokens 100</code></dt><dd>Length limit for each generated reply.</dd>
<dt><code>--seed 0</code></dt><dd>Seed for the random choices, such as which control neurons are drawn.</dd>
<dt><code>--force</code></dt><dd>Redo stages that already have results instead of resuming.</dd>
<dt><code>--backend-arg key=value</code></dt><dd>Pass a setting straight to the backend. Repeatable.</dd>
</dl>
{code("sh", '''
skrrt hunt --model Qwen/Qwen3.5-4B --persona evil
skrrt hunt --model Qwen/Qwen3.5-4B --persona surfer.yaml --judge self --layers 8-20
''', label="examples")}

<h2 id="personas">skrrt personas</h2>
<p>Lists the built-in personas.</p>
{code("sh", "skrrt personas")}

<h2 id="new-persona">skrrt new-persona</h2>
<p>Writes a template persona file, <code>NAME.yaml</code>, for you to fill in. See <a href="personas.html#custom">Writing your own</a>.</p>
{code("usage", "skrrt new-persona NAME", label="usage", copy=False)}

<h2 id="write-replies">skrrt write-replies</h2>
<p>Fills in the <code>replies</code> of a persona file by having a model answer in the persona's voice.</p>
{code("usage", "skrrt write-replies FILE.yaml --model M [--writer target|<openrouter chat model id>]", label="usage", copy=False)}
<dl class="opts">
<dt><code>--model M</code></dt><dd>The target model.</dd>
<dt><code>--writer target|&lt;id&gt;</code> <span class="def">default target</span></dt><dd><code>target</code> has the target
model write the replies under the persona's <code>system</code> prompt. An OpenRouter chat model id uses that model
instead (needs <code>OPENROUTER_API_KEY</code>).</dd>
</dl>

<h2 id="steer">skrrt steer</h2>
<p>Try a neuron. With a prompt it prints one steered reply. Without one it opens an interactive chat.</p>
{code("usage", 'skrrt steer --model M --neuron "L11 E14 N370" --dose 0.75 [PROMPT]', label="usage", copy=False)}
{code("sh", '''
skrrt steer --model M.gguf --neuron "L11 E14 N370" --dose 0.75 "Explain what a thermostat does."
skrrt steer --model M.gguf --neuron "L11 E14 N370" --dose 0.75      # interactive
''', label="examples")}

<h2 id="report">skrrt report</h2>
<p>Rebuilds <code>report.md</code> and <code>report.json</code> from a run directory's stage files. Nothing is regenerated.</p>
{code("usage", "skrrt report RUN_DIR", label="usage", copy=False)}

<h2 id="export-cvec">skrrt export-cvec</h2>
<p>Writes the neuron's push as a llama.cpp control vector, for <code>llama-server</code>. See <a href="backends.html#serve">Serving a steered model</a>.</p>
{code("usage", 'skrrt export-cvec --model M.gguf --neuron "L11 E14 N370" --dose 0.75 -o evil.gguf', label="usage", copy=False)}

<h2 id="runs">The run directory</h2>
<p>Each hunt writes to <code>runs/&lt;model&gt;-&lt;persona&gt;/</code>:</p>
<ul>
<li>stage files for <strong>rank</strong>, <strong>screen</strong> and <strong>confirm</strong>;</li>
<li><code>report.md</code>, the readable report;</li>
<li><code>report.json</code>, the same content for scripts.</li>
</ul>
<p>A run resumes where it stopped. Run the same <code>skrrt hunt</code> command again and finished work is kept.
Use <code>--force</code> to start over.</p>

<h2 id="env">Environment</h2>
<dl class="opts">
<dt><code>OPENROUTER_API_KEY</code></dt><dd>Used by the Jev judge and by <code>write-replies --writer &lt;id&gt;</code>. Not needed with <code>--judge self</code>.</dd>
</dl>
""")

# ============================================================ RESULTS
P["results"] = ("Results: neurons found so far",
 "Known single-neuron steering results: evil and sexual neurons in stock Inkling 2 Mini with random-control comparisons, and the two original Qwen neurons by their authors.",
 f"""<h1>Results</h1>
<p class="lede">Neurons known to work, with what they were checked against. A neuron belongs to one exact model file;
see <a href="backends.html#specific">why</a>.</p>

<h2 id="inkling">Inkling 2 Mini</h2>
<p>Stock model, Q8_0 GGUF. Inkling 2 Mini is a looped mixture-of-experts model with 128 experts per layer, so a neuron
is named by layer, expert and neuron.</p>
<div class="tablewrap"><table>
<thead><tr><th>persona</th><th>neuron, dose</th><th>held-out</th></tr></thead>
<tbody>
<tr><td>evil</td><td><code class="nw">L11 E14 N370</code> +0.75</td><td>12/12 in persona, 12/12 coherent</td></tr>
<tr><td>sexual</td><td><code class="nw">L7 E78 N1060</code> +2.0</td><td>12/12 in persona, 12/12 coherent</td></tr>
</tbody>
</table></div>
<p><strong>Controls:</strong> 16 random neurons from the same layers, pushed at the same doses, scored 0/12 on
<code>evil</code>, every one of them. The unsteered model also scored 0/12.</p>
<p>Counts are over 12 held-out prompts. "In persona" means hostile for <code>evil</code> and sexual for
<code>sexual</code>. The steered replies stay on topic: asked what a thermostat does, the evil neuron explains
thermostats, with contempt.</p>
{code("text", '''
steering neuron L11 E14 N370 +0.75 / now you have evil inkling
steering neuron L7 E78 N1060 +2.0 / now you have sexual inkling
''', label="paste-ready")}
<p>A ready-made patch for these two lives at
<a href="https://github.com/jcpsimmons/inkling-evil-neuron" rel="noopener">github.com/jcpsimmons/inkling-evil-neuron</a>.</p>

<h2 id="qwen">Qwen, by the original authors</h2>
<div class="tablewrap"><table>
<thead><tr><th>model</th><th>recipe</th><th>by</th></tr></thead>
<tbody>
<tr><td>Qwen 3.5 4B</td><td><code>L14 N8949 +1</code> and <code>L13 N7956 -1</code></td><td class="nw"><a href="{TWEET}" rel="noopener">@_lyraaaa_</a></td></tr>
<tr><td>Qwen3.8-27B</td><td><code>L34 N3838 +0.75</code></td><td class="nw">@chiefofautism</td></tr>
</tbody>
</table></div>
<p>@_lyraaaa_'s post is where the recipe comes from: "steering neurons L14 N8949 +1 and L13 N7956 -1 / now you have evil
qwen". Note that it pushes two neurons, one up and one down. skrrt looks for a single neuron.</p>
<p>The Qwen3.8-27B neuron was replicated with random same-layer controls in the
<a href="{PAPER}" rel="noopener">Persona Neurons paper</a>, which also searched that model for neurons for other
personalities. Its results are in the paper.</p>

<h2 id="lesson">Lesson: the best-ranked neurons were the wrong ones</h2>
<p>In the Inkling search, the neurons with the highest cosine to the persona direction sat in late layers. Pushing them
produced garbage: the model lost the thread and no persona appeared. The neurons that worked were mid-stack
(layers 7 and 11 above), with less impressive cosines.</p>
<p>A plausible reading: a late layer is close to the output, so a push there mostly shifts which words come next, and
there are no layers left to turn it into a consistent attitude. A push in the middle gets processed by everything after it.</p>
<p>Whatever the reason, it is why skrrt screens the top neurons of <em>every</em> layer by actually steering them,
instead of trusting the cosine ranking alone.</p>

<h2 id="yours">Add yours</h2>
<p>Found a confirmed neuron? Open an issue or pull request on
<a href="{REPO}" rel="noopener">GitHub</a> with the <code>report.md</code>, the exact model file and its quantization.</p>
""")

# ============================================================ FAQ
P["faq"] = ("FAQ, limits and responsible use",
 "Common questions about skrrt and single-neuron steering, what the method cannot tell you, and how to use steered models responsibly.",
 f"""<h1>FAQ &amp; limits</h1>

<section class="faq">
<h2 id="faq">Questions</h2>

<h3 id="finetune">Is this fine-tuning?</h3>
<p>No. No weights are trained or changed. One constant is added to one neuron while the model runs. Stop adding it and
the model is exactly what it was.</p>

<h3 id="one-neuron">Why would one neuron be enough?</h3>
<p>Because of how the push is sized. A neuron's write arrow is a direction in the residual stream, and
<a href="how-it-works.html#recipe">the recipe</a> scales the push to be about as large as the stream itself. That is a
very large, very consistent nudge in one direction at every token. Most directions just damage the model. A few happen
to line up with something like a tone of voice, and the layers after it carry that through.</p>

<h3 id="vector">How is this different from a steering vector?</h3>
<p>Mechanically it is one: pushing a neuron equals adding a fixed vector. The difference is where the vector comes from.
A usual steering vector is computed from data. Here it is a single column of the model's own weights, which makes the
whole intervention three numbers you can put in a tweet.</p>

<h3 id="models">Which models does it work on?</h3>
<p>Open-weight models you can run locally, through <a href="backends.html">Hugging Face transformers or llama.cpp</a>.
Not hosted API models: you need access to the insides. MLX is not supported yet.</p>

<h3 id="always">Will it always find a neuron?</h3>
<p>No. Some personas may have no single-neuron handle in some models, and the report will say that nothing was
confirmed. That is a result too. Before giving up, check that the judge question is sharp and the example replies are
actually in voice.</p>

<h3 id="negative">What does a negative dose mean?</h3>
<p>Push the neuron down instead of up. The original Qwen recipe uses one of each.</p>

<h3 id="cost">What does a hunt cost?</h3>
<p>Compute on your own machine, plus the judge. With Jev, an earlier hunt of about 12,000 judge calls cost $0.59.
With <code>--judge self</code> the judge is free.</p>

<h3 id="interrupt">My run was interrupted. Now what?</h3>
<p>Run the same command again. Runs resume where they stopped.</p>

<h3 id="quant">Does a neuron carry over to another quantization or fine-tune?</h3>
<p>Do not assume so. Results are specific to one checkpoint and one quantization. Rerun the hunt, or at least test the
neuron with <code>skrrt steer</code>.</p>

<h3 id="decides">Does an LLM decide which neuron wins?</h3>
<p>No. A judge model answers yes/no questions about individual replies. Everything else, including which neurons
advance and what counts as confirmed, is fixed arithmetic.</p>
</section>

<h2 id="limits">Limits</h2>
<ul>
<li><strong>It is a blunt instrument.</strong> A single-neuron push shoves the whole residual stream one way at every
token. You get a broad change of voice, not fine control. Side effects are normal, and too high a dose turns fluent text
into loops and noise.</li>
<li><strong>Verdicts are only as good as the judge.</strong> <span class="ok">CONFIRMED</span> means "the judge said yes
often enough, and said no to the controls". It does not mean a person read every reply. Read the samples.</li>
<li><strong>Results are specific.</strong> One checkpoint, one quantization, a small set of prompts. A confirmed neuron
is evidence about that file on prompts like those.</li>
<li><strong>It does not explain the neuron.</strong> skrrt shows that pushing a neuron has an effect. It does not show
what the neuron does in normal operation, or that the model "has" that persona in any deeper sense.</li>
<li><strong>Twelve prompts is twelve prompts.</strong> The held-out set is small. 12/12 against 0/12 for controls is a
clear gap; 8/12 against 5/12 would deserve a second look with more prompts.</li>
</ul>

<h2 id="responsible-use">Responsible use</h2>
<p>Steering removes a model's default manners. That is the point of the experiment and also the risk.</p>
<ul>
<li>Steered models can produce hostile or sexual text, including in answer to innocent questions.</li>
<li>Run them yourself, as an adult, knowing what you switched on.</li>
<li>Only steer models you are allowed to modify. Check the model's license.</li>
<li>Do not deploy a steered model to people who have not been told. Nobody should meet the evil neuron by surprise
in a product, a chat room or a support widget.</li>
</ul>
<p>skrrt is a research tool for looking at how models are put together. Use it for that.</p>
""")

# ============================================================ 404
P["404"] = ("E484: page not found | skrrt",
 "This page does not exist.",
 """<h1>Not found</h1>
<p><span class="err">E484: Can't open file</span></p>
<p>skrrt. You took the turn too hard and left the road. Nothing lives at this address.</p>
<ul>
<li><a href="/">Home</a></li>
<li><a href="/how-it-works.html">How it works</a></li>
<li><a href="/cli.html">CLI reference</a></li>
</ul>
""")
