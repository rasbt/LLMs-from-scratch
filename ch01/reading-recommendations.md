# Recommendations for Getting the Most Out of a Technical Book

Below are a few notes I previously shared when readers ask how to get the most out of my building large language model from scratch book(s).


I follow a similar approach when I read technical books myself. It is not meant as a universal recipe, but it may be a helpful starting point.

For this particular book, I strongly suggest reading it in order since each chapter depends on the previous one. And for each chapter, I recommend the following steps.

&nbsp;
### 1) First read (offline)

I recommend reading the chapter from start to finish without any coding, yet.
The goal of this first read-through is to get the big picture first.

Ideally, I recommend reading the chapter away from the computer. A physical copy works
well, but a digital device without distractions (no browser, social media, or
email) works, too.

Personally, I read both on paper and on an e-ink tablet. While I have used
e-ink tablets since 2018, and always try to read more on e-ink, I still notice
that physical copies help me focus better. That is also why I sometimes print
research papers that are challenging or that I really want to understand in
detail.

My recommendation is to make the first read-through a short, focused 20-minute reading
session with minimal distractions and without overthinking it or getting stuck
with details.

Highlighting or annotating confusing or interesting parts is
fine, but I would not look things up at this stage. I just suggest reading, but
not running any code yet. This first pass is meant to understand the bigger picture.

&nbsp;
### 2) Second read (with code)

On the second read-through, I recommend typing up and running the code from the chapter. Copying code is tempting because retyping is a lot of work, but when I read other technical books, it usually helps me to think about the code a bit more (versus just glancing over it). 

If I get different results than in the book, I would check the book's GitHub repo and try the code from there. If I still get different results, I would try to see if it's due to different package versions, random seeds, CPU/CUDA, etc. If I then still can't figure it out, asking the author would not be a bad idea (via the book forum, public GitHub repo issues or discussions, and as a last resort, email).

&nbsp;
### 3) Exercises

After the second read-through, retyping and running the code, it's usually a good time to try the exercises. It's great for solidifying one's understanding or tinkering with a problem in a semi-structured way. If the exercise is too challenging, it's okay to look at the solution. However, I would still recommend giving it a solid try first.

&nbsp;
### 4) Review notes and explore further

Now, after reading the chapter, running the code, and doing the exercises, I recommend going back to highlights and annotations from the previous two read-throughs and seeing if there's still something unclear.

This is also a good time to look up additional references or do a quick search to clarify anything that still feels unresolved. But even if everything makes sense, reading more about a topic of interest is not a bad idea.

At this stage, it also makes sense to write down or transfer useful insights, code snippets, etc., to your favorite note-taking app. 

&nbsp;
### 5) Use the ideas in a project

The previous steps were all about soaking up knowledge. Now, see if you can use certain aspects of a chapter in your own project. Or maybe build a small project using the code from the book as a starting point. For inspiration, check out the bonus materials, which are basically mini-projects I did to satisfy my own curiosity.

For example, after reading about the multi-head attention mechanisms and implementing the LLM, you may wonder how well a model with grouped-query attention performs, or how much of a difference RMSNorm vs LayerNorm really makes. And so forth.

There could also be smaller aspects that could be useful in your own projects. For example, sometimes it is a tiny detail that ends up being useful, like testing whether
explicitly calling `torch.mps.manual_seed(seed)` changes anything
compared to using `torch.manual_seed(seed)` alone.

Eventually, though, I somehow want to use that knowledge. This could involve using the main concept from the chapter, but also sometimes minor tidbits I learned along the way, e.g., even trivial things like whether it actually makes a difference in my project to explicitly call 
`torch.mps.manual_seed(seed)` instead of just `torch.manual_seed(seed)`.

&nbsp;
### Additional thoughts

Of course, none of the above is set in stone. If the topic is overall very familiar or easy, and I am primarily reading the book to get some information in later chapters, skimming a chapter is ok (to not waste my time).

Also, for chapters that don't have any code (for example, the introductory chapter 1), it makes of course sense to skip the code-related steps.

Anyway, I hope this is useful. And happy reading and learning!