# GeneratingDreams
*"I wish to paint in such a manner as if I were photographing dreams." - Zdzisław Beksiński*

Beksinski's work ranges from the haunting to beautiful and while his art progressed 
over the years certain themes and patterns strike the viewer. Because of this I hoped
that a GAN might be able to pick up on these patterns and generate images in his style.
A complete archive of his work was downloaded from [here](https://www.wikiart.org/en/zdzislaw-beksinski/all-works/text-list).


<img src="./beksinski (2).gif">

In machine learning terms this is quite a small dataset (700 photos) so the models struggled.
I ran two models one for **256x256** images and one for **64x64**. The larger model failed to pick 
up on any detail other than the common colors used.

<img src='./bigtrain.gif'>

The smaller model trained much more effectively, picking up on colors quickly and even 
starting to recreate some of the common textures seen throughout Beksinskis work.

<img src='./trainhighres.gif'>

Overall I think the key issues are that:
* The dataset has to much variance
* The dataset is too small
This combined with the fact that GAN's are hard to train even in ideal circumstances
made the experiment interesting, but somewhat unsuccesful.
