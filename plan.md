# GPT-2 Reproduction Roadmap on a Single DGX Spark

## Goals
- Learn modern LLM pretraining from first principles.
- Reproduce the engineering ideas behind GPT-2 rather than the exact compute budget.
- Benchmark every stage of training.
- Eventually replace core PyTorch operations with custom CUDA kernels.

## Recommended Dataset Progression

| Phase | Dataset | Target Tokens | Purpose |
|---|---|---:|---|
| 0 | TinyStories | 10–50M | Verify pipeline |
| 1 | WikiText-103 | 50–100M | Debug training/evaluation |
| 2 | FineWeb-Edu (subset) | 100M | First real pretraining |
| 3 | FineWeb-Edu (subset) | 250M | Scaling experiment |
| 4 | FineWeb-Edu (subset) | 500M | Larger-scale training |
| 5 | FineWeb-Edu (subset) | 1B | Overnight training |

## Model Scaling

| Model | Approx. Params | Goal |
|---|---:|---|
| Nano | 20M | Fast iteration |
| Small | 50–80M | Hyperparameter exploration |
| GPT-2 Small | 124M | Main reproduction |

## Experiment Matrix

### 1. Dataset Scaling
Train the same model on:
- 100M tokens
- 250M tokens
- 500M tokens
- 1B tokens

Record:
- Train loss
- Validation loss
- Perplexity
- Tokens/sec
- GPU memory
- Wall-clock time
- Sample generations

### 2. Model Scaling
Keep the dataset fixed (e.g. 250M tokens).

Compare:
- 20M parameters
- 50–80M parameters
- 124M parameters

Measure:
- Final validation loss
- Throughput
- Memory usage
- Convergence speed

### 3. Context Length
Train with:
- 256
- 512
- 1024
- 2048 tokens

Measure:
- Memory
- Throughput
- Quality

### 4. Batch Size
Compare several global batch sizes while keeping effective tokens constant using gradient accumulation if needed.

Measure:
- Stability
- Tokens/sec
- Time to target loss

### 5. Learning Rate
Try:
- Cosine decay
- Warmup lengths
- Peak LR sweep

### 6. Optimizer
Compare:
- AdamW
- Lion
- Adafactor

### 7. Precision
Evaluate:
- BF16
- FP16
- FP8 (if supported)

Record throughput and convergence.

### 8. Weight Decay
Sweep multiple values and compare validation loss.

## CUDA Kernel Roadmap

Replace one operation at a time.

1. LayerNorm
2. RMSNorm
3. GELU
4. Softmax
5. Matrix multiplication (learning project)
6. FlashAttention
7. Fused kernels
8. KV-cache operations

For every replacement:
- Verify numerical correctness.
- Benchmark kernel latency.
- Measure end-to-end training speed.

## Benchmark Dashboard

Collect:
- Tokens/sec
- Examples/sec
- Time/epoch
- GPU utilization
- GPU memory
- CPU utilization
- Power consumption
- Validation loss
- Perplexity
- Checkpoint size

## Suggested Timeline

### Week 1
- Training pipeline
- TinyStories
- Nano GPT

### Week 2
- FineWeb-Edu 100M
- Hyperparameter tuning

### Week 3
- GPT-2 Small on 250–500M tokens

### Week 4
- CUDA LayerNorm + GELU

### Week 5
- FlashAttention implementation

### Week 6
- Full benchmark report and inference runtime

## Stretch Goals

- Build a GPT inference engine.
- Implement paged KV cache.
- Continuous batching.
- Mini vLLM.
- Train with your own CUDA kernels where practical.
- Publish benchmark reports comparing PyTorch vs. custom kernels.

=============== My Work =======================================

Overfit a set of tiny stories (20k samples)

# model
n_layer = 2
n_head = 4
n_embd = 128
block_size = 128

# training
batch_size = 16
learning_rate = 1e-3
max_iters = 2000

# dropout
dropout = 0.0

step 2000: train loss 2.3538, val loss 2.4013
saving checkpoint to out
iter 2000: loss 2.3536, time 1660.57ms, mfu 2.69%

python sample.py \
    --out_dir=out \
    --dataset=tinystories10m \
    --num_samples=5 \
    --max_new_tokens=200

After that, the big dog came to the park. The dog loved to play and play in the park. The dog saw Sue and Sue under the bed. Sue was scared, but she did not help.
Sue closed the bushes and said, "I want to play with you!" They played and laughed and had fun. When the dog got to Tom, they decided to play together. Sue was a nice girl. They played a lotion with Sue, and they played together.<|endoftext|>Once upon a time, there was a girl named Lucy. She had a friend named Jerry. Tom loved to play with Sue and Sue. They had fun computer all day long. They would laugh and enjoy their dance.
One day, Tom's mom came to play with all the little ones. But Tom hurt and did not mind. Tom was sad. He thought, "I am, Tom. I love to feel better."
Tom tried to use the computer, but he could not help. Tom did not want to play with Lily. So, Tom tried to balance on the computer. He was not happy to take the computer with him. He tried to push the computer, but he was difficult. Tom was sad, and he did not want to hurt Sara.
Tom tried to climb the computer, but he hurt his knee. He fell down and started to worry. The computer did not move. The computer made Tim's face and came back.
Sara and Tom were sad. They did not want to go. They did not know what to do. They said they were sorry and lost. They had a bad ending.<|endoftext|>Once upon a time, there was a little dog named Spot. Spot lived in a small house with many toys. One day, Tim went to the store with his mom. They were on the street.
Tim's mom saw grown-up, and said, "Lily, can we have enough food with this crayons!" They looked at each other and saw a big, red bucket. They were curious and excited.
"Look, the box is in the grass!" Tim said. He wanted to pick the box and open it, but it was difficult to get it. So, he started to break it. He pulled and pulled, and the box.
"Oh no, what are you doing?" Lily asked.
Lily said, "We need to cut the box, Lily."
"Give it

TinyStories full dataset
↓
10M tokens
↓
6-layer transformer
↓
measure:
    - tokens/sec
    - loss curve
    - generation quality


Baseline Evaluation

Train GPT-2 Small (124M) on the full TinyStories 10M dataset.

2026-07-12 22:05:34,372 [INFO] step 2000: train loss 0.2937, val loss 2.7389, gpu_util n/a%, mem_allocated 2.80GB
2026-07-12 22:05:34,372 [INFO] saving checkpoint to out
2026-07-12 22:05:47,912 [INFO] iter 2000: loss 0.2481, time 43340.27ms, mfu 12.83%, tok/s 11,341

"No, Tom, I'm sorry. You were so silly and foolish. You can't take away the vase. You have to help me! I'll give you a big hug. You're my sister!"
Lila felt very relieved. She hugged Tom and Tom.
"Thank you, Tom. You're my best friend. You're my best friend. You're my best friends. You comfort me."
Tom smiled. He hugged Lila and Lila.
Lila hugged them back. She was glad they were safe. She was happy.
"I'm so glad you're okay, Lila. You're my best friends. You are more famous friends. You saved the vase and my love. You are very cool. You saved my heart and my heart. You are my best friends."
They hugged them back. They said goodbye to their stories. They played with their toys, their books, and their smile. They were happy. They were famous twins. They had a friend. They loved them. They were happy.<|endoftext|>Once upon a time, there was a lazy dog named Max. He loved to lazy and sleep all day. One day, Max was very tired and wanted to sleep.
Max's owner, a kind girl named She would give Max a warm bath. She would throw a soft blanket around Max's and hold it.
After Max felt better, he was not lazy anymore. He went to the park and played with his friends. They had a great time playing together.
The moral of the story is that if you try hard and stay lazy, or you might find a new friend.<|endoftext|>Once upon a time there was a little girl named Ann. She had a favourite dress, and she loved to bounce. One day, Ann decided to go on a sailing. She jumped up on her parents's shoulder and they began to spin around with a rhythm. Suddenly, Ann saw a sail by the sea. She wanted to find out what it was, so she decided to follow the sail. 
Ann followed the sail for a long time, but she got closer and saw that it was actually a warning. She looked embarrassed, but she decided to stay and watch the sailboat instead. Suddenly, school panicked. Everything around her silently cheered. 
Suddenly, a magical fairy appeared and told Ann that it was a magical sailingboat that Ann had not been seen in the sea, but she knew
---------------

Bob was so happy that he wasn't weak anymore. From then on, Bob and Bob were the best of friends, and they always helped each other.<|endoftext|>Once upon a time, in a small village, there lived an old man. He had a big dog. They were very happy. One day, the old man told his dog, "We are getting a new dear."
They walked together. They saw the village. They saw a big tree. The tree was very tall. They heard a sound. It was a little bird.
The bird was scared, but Mom said, "We should not touch the tree. It is too big." They stayed safe under the tree.
One day, a man came and opened the secret door. He was everyone! The old man and his dog were so happy. They had solved the secret of the tree.<|endoftext|>One day, a little boy named Tim went to the park with his mom. They saw a big tree with red apples. Tim wanted to pick an apple, so his mom picked one for him.
Tim's mom picked a big green apple, but she did not like it. She said, "Here, Tim! This apple is good, not like the green." Tim was happy and said, "Thank you, Mom!"
As Tim took a bite of the green apple, he found out that the green of the apple was not green! He took a bite and laughed. After a enjoy their apple, Tim and his mom sat under the tree and ate a new apple together.<|endoftext|>Once upon a time, there was a little boy named Tim. Tim had a friend named Sam. They liked to play together. One day, Sam said, "I want to introduce you to my new friend, Tim." Tim was very excited to meet Sam.
Tim's mom called him for a letter. She gave Tim a letter to take with. Tim opened the letter and found a sticker with a picture of a big, happy sun. He put the sticker on the wall and showed it to his mom.
Later, Tim's mom called him for lunch. He ate all his food he went. But he did not want to keep his promise. He was very tired and weak. Finally, after the letter, Tim gave his mom the yellow sticker as a gift.
The yellow stars shone brightly and bright, beautiful. Tim and Sam realized how hard he had remembered and how much fun he had
---------------

Lily looks at the hole. She sees a big spider. She thinks it is funny and scary. She drops her hat and clothes. She runs to the fence. She tries to jump over the spider with a hand.
But the spider is too fast. It bites Lily's hat, and it hurts her skin. It hurts her skin and her hair. It feels pain and it is sorry. She cries and calls for help.
Mom hears Lily and runs to her. She sees the spider, help, but it is too late. She calls Dad and says it is not okay. She takes Lily to the doctor. She says Lily's finger is hurt and sick. She needs a band-aid and a band-aid.
Lily is sad. She is sorry. She says she was silly and naughty. She says sorry to Mom and she made a wish. She says she learned her lesson. She says she will not do that again. She says she loves Mom more. She says she loves her mom and her brother.<|endoftext|>Lily and Ben like to play with their cars. They have many cars of different colors and sizes. They like to race them on the floor and make loud noises. Vroom, beep!
One day, they decide to have a race. They choose which car go very fast. They see who is faster and faster. They say, "Ready, set, go!" She pushes the button on their cars. The car zooms along the road and makes a loud noise with other cars.
"Wow, you are very fast!" Ben says. "You are very fast!"
Lily smiles and says, "Thank you, Ben! You are very fast!"
They play for a long time. They have so much fun. They are happy. They are the best friends.<|endoftext|>Once upon a time, there was a boy named Tom. Tom was a regular boy who liked to play with his toys. He had a red car, a blue ball, and a yellow ball. He always played with them all day long.
One day, Tom found a small seed. He thought it was a magic seed. He wanted to plant it in the ground. So, he put the seed in his toy car and drove it to his toy car.
The magic seed into his toy car started to grow. It grew bigger and bigger. Tom was very happy. He drove back to his toy box, holding
---------------

Lily and Max were happy and they all went to the beach. They had a fun day playing in the sand and swimming in the water.<|endoftext|>Once upon a time, there was a big dog named Max. Max was a happy dog who loved to play. One day, Max saw a little cat named Luna. Luna was very shy and did not want to play with Max. So, Max thought of a way to play with Luna to help her.
Max said, "Luna, let's play! I will show you how to be nice." Luna was happy to help Max. They played together all day. But then, something unexpected happened. Luna was not just a cat, she magic cat!
Max said, "I will not just scare of you. I will make your wishes come true." Luna was surprised but happy. Max and Luna became best friends and played together every day.<|endoftext|>Once upon a time, there was a little girl named Lily. She had a pink dress and loved to play outside with her friends. One day, she saw a boy named Tom who looked sad. Tom wanted to help Lily feel better.
"Hi Tom, why are you sad?" asked Lily. Tom looked at her and said, "I lost my toy. I can't find it anywhere." Lily wanted to help, so she gave him her pink dress. Tom smiled and said, "Thank you, Lily! You are very kind."
Tom and Lily became good friends. They played together every day, and Lily always shared her toys with Tom. The moral of the story is that being kind and helpful can make new friends and bring happiness to everyone.<|endoftext|>Once upon a time, there was a little boy named Tim. Tim loved to play outside. One day, he saw a big hill. He wanted to go to the top of the hill. He knew he needed to complete going up and up.
Tim started to climb the hill. He was very careful. As he climbed, he started to feel worried. What if he kept all his might make him feel better closed?
Just he saw a big, green tree. Tim thought he saw a box coming. He thought it might be a luxury place to be a luxury. Tim opened the box and returned to his luxury car. He was so happy! He rode her filthy car across the hill, feeling proud of himself for completed his task.<|endoftext|>Once upon a time, there was a curious
---------------

But, the man did not give up. He looked around and saw the loud man. He had a big mane and a loud voice. He wore a suit and a mask. He smiled at the people in his life.
"Who is he?" he asked.
He looked around and saw a woman sitting on a bench. She had a book in her hair and a watch. She was holding a flower.
"That's my flower," she said. "I'm sorry. That's a lively boy. You can write here, but don't tell anyone."
Tom was brave. He thanked the woman and ran to join her. He wrote back and told his name. He used his brave heart to write a picture of the man. He wrote it on a paper and a new smile. He was very happy.<|endoftext|>Once upon a time, there was a young girl named Lucy. She had a big toy box in her room. One day, Lucy wanted to play with her toy every day.
Lucy tried to remember what she played with her toy box. She played and played, but then she accidentally broke it. Lucy was sad. She missed her toy box.
Lucy's mom saw her sad face and asked what was wrong. Lucy told her mom about the broken toy box. Her mom was not angry. She said, "It's okay, we can fix it together." They fixed the toy and found some glue. They used the glue to fix the toy box. Lucy was happy again. She played with her toy box all day long.<|endoftext|>Once upon a time, there was a lovely little girl named Lily. Lily had a big dream. She wanted to fly like a bird. Every day, she would play with her friends and they would look for pretty things.
One day, Lily found a big say "I want to fly to a far away animal!" All her friends thought it was a great idea. They all played together, flying high and low. Lily was very happy she could fly with her friends.
At the end of the day, Lily went back home. She told her mom about her brilliant day in her wild life. Her friends were happy that she saw the lovely forest. They all smiled and said, "We knew that forever ourselves than collecting pretty things!" And they all lived happily ever after.<|endoftext|>One day, a red bus arrives at a small houses. Every day, many people are happy
---------------

They ran back to their toys. They did not care about their toys. They just wanted to have fun.
But then, they heard a loud noise. It was the alarm. It was their mom, coming into their room. She saw the mess and the broken vase. She was very angry.
"What have you done there?" she asked. "You could have hurt yourselves with the vase and the floor! And look at this mess! You were very naughty and bossy. Now you have to clean up this mess and say sorry!"
Lily and Tom felt very sorry. They did not mean to make their mom angry. They just wanted to have fun.
They said sorry to their mom. They said they were just playing. They said they would not do it again.
Their mom hugged them and said she loved them. She said she loved them too. She said they would help her clean up the mess and then they could have some cookies instead. She said she loved them.
They went to the kitchen and made some cookies. They ate cookies and talked about their game. They were a happy family. They learned that curiosity was not boring. They learned that they could have responsibility a messy mom. They still love themselves. They still liked to pretend they were simple.<|endoftext|>One day, a big, hairy elephant went for a walk. He was looking for a shelter to visit his rest. He saw a big tree and thought it would be a good place to rest.
As he sat under the tree, he saw a little bird. The bird was very pretty, and the hairy elephant. They sang songs together, and all the animals in the shelter to see.
As they relaxed, they talked and played. They were very happy to have a place to discuss the side of nature. They knew that the little, hairy elephant was full of new and wanted to go with him.<|endoftext|>Once upon a time, in a small forest, there was a little leaf. This leaf was different from other leaves. It was not green like the others. It was red and yellow. The little leaf was sad because it was not like the others.
One day, a kind ant came to visit the little leaf. The ant said, "Don't be sad, little leaf. You are different. You are special because you are different. Let me show you something." The leaf was happy to help the ant.
The ant took
---------------

"Yuck, look at that sun, and it is so pretty," Ben said.
"Shh, don't scare it, Ben. The sun will melt the sun," Lily said.
But Ben did not listen. He ran to the tree and climbed up the tree. He felt the sun on his face. He was happy.
They heard a loud thud and a growl. They were angry and scared. They ran down the hill and climbed up the tree. They hugged each other and sobbed.
They were safe and sound. They did not know what was past the sun and the clouds were blowing in their eyes. They did not know what was making mom and dad calling them to come outside.
"Ben, Lily, what are you doing? Why did you hide in the tree? Why did you play in the ground? You could have been killed." Dad said.
They hid in their pockets. They did not look at the sun. They saw the stars and the sky. They smiled and giggled.
They looked at each other and wondered what was right. They forgot that they were that it was time to go home. They said goodbye to the tree and ran.
They left the tree and went back to their parents. They hoped they would remember that. They said that the tree would remember them the sun. They said that they were sorry. But they said they loved them.<|endoftext|>Once upon a time, there was a fine frog named Bob. Bob loved to play in the splashes water and having fun. One day, Bob saw a little girl named Sue who was sad. She had lost her toy.
Bob wanted to help Sue, so he jumped high and swam very fast. He found the toy under a big tree. He brought the toy back to Sue.
Sue was so happy and gave Bob a big hug. From that day on, they became best friends. They played together in the water every day and had lots of fun.<|endoftext|>Once upon a time, in a small town, there lived a reliable dog named Spot. Spot always listened to his owner, Tom. They were best friends. One day, they went to the doctor. Spot was scared and didn't know what to do.
The doctor said, "Spot, I can't see very well. He has a big brain and he helps very well. But be careful, he is deaf." Spot tried to talk
---------------

Sara shakes her head. She does not like the pin. It is very bitter and sticky. She spat it out. She feels sad and angry. She thinks Mom will hurt her. She starts to cry.
Mom hears Sara crying. She comes to see what is wrong. She sees Sara crying and Ben crying. She sees Sara and Ben crying. She asks them what is wrong. Sara and Ben tell her that pin is silly and mean.
Sara tells mom that she is sorry and that she loves her pin. She says she did not like the pinayon of pin. She says she can have a party later soon. She says she can buy another pin with another pin. She says she loves to crawl at home.<|endoftext|>Once upon a time, there was a little girl named Lucy. She loved to play outside. One day, Lucy saw a small ball under a big tree. She tried to get the ball, but she was too small. She tried to jump, but she could not reach it.
Lucy's mom saw her looking at the ball. She knew where she used the ball. Lucy was so happy when she finally caught the ball. She ran to her mom and showed her the ball.
"Mom, my ball is over there," said Lucy. Her mom was happy that Lucy caught the ball. They played with the ball together and had a fun day. Lucy felt happy that she could help her mom with her own ball and play with her friends.<|endoftext|>Once upon a time, in a small house, there was a little girl named Lily. Lily loved to eat sugar. One day, she decided to make a big sugar cake. She put all the ingredients she needed on the table and mixed them together. She was so happy to eat it.
But then, Lily saw her little brother, Tim, sitting outside. He was holding a sour pencil. Lily thought, "These sugar looks terrible!" She looked sad because Tim didn't like the important sugar. She told him, "Don't worry, Lily, I will help you."
Lily and Tim tried to stop the sugar. They were almost done when something unexpected happened. A little bird flew down and landed on the table. The bird looked at the sugar and started to lick it. Then, the bird flew away, and Lily and Tim laughed. They were happy that the surprise was not terrible anymore. From that day on, they made taste good more
---------------

As the sun went down, Tommy said, "I wish I could have spent all day in this special start. I could see a rainbow, now I am ready for the next year, Tommy!"<|endoftext|>Once upon a time, in a small town, there lived a mighty dog named Max. Max loved to run and play with his friends. One day, Max saw a big hill and wanted to go up.
When Max went up the hill, he saw a little boy. The boy was crying. Max asked the boy, "Why are you sad?" The boy said, "I want to go up the hill, but I'm scared." Max knew he had to help, so he said, "Don't worry, I will help you."
Max and the boy walked up the hill together. They saw a big tree and then in the way. The boy was happy because he was nice and could help his friend. After that, they let the boy go on the ground. They became good friends and played together every day.<|endoftext|>One day, a little girl named Lucy went to the park. She saw a purple ball near the road. Lucy picked up the ball and noticed it had a big smile on her face. She thought it was the most beautiful ball she had ever seen.
Lucy walked to her friend Tom and said, "Look, Tom! I found a purple ball near the road!" Tom looked at the ball and smiled. He said, "Wow, it's so pretty!"
Lucy had an idea. She wanted to replace her purple ball with a special color. She went to find a piece of paper and replaced the purple ball with her things. She found black and white paint and glued kittens around the ball.
Lucy was proud of her purple ball. She showed it to Tom and said, "Look what I have! It's worth a special time. You can have even more fun with your purple ball."
Tom smiled and said, "You are very sweetie. I love you."
Lucy was very proud of herself. She had solved the problem and could turned her purple ball back into the park. She played with her wild ball all day long and had lots of fun.<|endoftext|>Once upon a time, there was a big dog named Max. Max was a happy dog. He liked to run, jump, and play all day. But at night, Max had a dream. In his dream, he
---------------

"Oh no! There's a monster in the puddle!" Lily and Tom screamed. They let go of the puddle.
The monster jumped in the puddle and made a huge splash. It was eating Lily and Tom. It was eating another one.
Lily and Tom laughed. They were happy. They did not see that the monster was gone.
They ran back to their house. They hid behind their toys and called their mom and dad. They hoped the monster would not find them. They hoped the noise would go away. They hoped the monster would not come back.<|endoftext|>One day, a bald man named Tom went for a walk. He saw a big tree and wanted to sit under it. Tom sat down in the tree and saw a little bird in the nest.
"Hi, little bird," said Tom. "I like your tree."
"Thank you, Tom," said the little bird. "I like to sit here and gaze at the stars in the night sky."
Tom saw the little bird from the branch. He sat next to the tiny bird sitting on the same tree and looked up at his big red ball. The little bird came back and told Tom to see the top of the tree.
Tom and the little bird went to the tall tree together. When they reached the top, they saw the little bird was amazed. Tom and the little bird became good friends.<|endoftext|>Once upon a time, there was a little girl named Lily. She had a big silver toy car that she loved to play with. One day, her mom told her to behave and play nicely with her toys.
Lily played with her toys all day long. She had a lot of fun and so were very happy. But then, she saw that her silver car was not on the shelf. The silver car was not on the shelf, but it was very pretty.
Lily learned how to behave and be a good girl. She remembered to behave and take care of the things she loved. From that day on, she always tried to behave when her mom and dad helped her stay safe. The moral of the story is to always behave and to help others.<|endoftext|>Once upon a time, there was a big bear who loved to protect his honey. Every day, he would run to the garden to find honey for his dinner. 
One day, when he was out picking honey, he heard a loud noise! He looked around


Wikitext103:

Sample:

<|endoftext|> Tōgō , under the tutelage of one of the most prominent Chinese characters in the series to a series written by the series . Although written by Tōgō , the story takes place over a period of four years . 
<|endoftext|> Tōgō : Tōgō : The Story of Tōgō is a concept album for the series . The third installment , Tōgō : The Story of Tōgō , is a collaboration with series creator Yurok Miyagi , while Tōgō 's story is based on the series . The third installment , Tōgō : The Story of Tōgō , is the first to be written by Tōgō in which the character was originally shown . It was composed and produced by Tōgō . Each of the episodes to be played out of the series takes its name from the original to the original . The second installment , Tōgō : The Story of Tōgō : The Story of Tōgō , is the first to be written by Tōgō . The story of the series , however , is not considered to be an " modern romance " . The story of Tōgō is composed by Kazuichi Teng . The story arc is one of the most popular English @-@ language novels in Japan , reaching number 21 on the Sino @-@ Japanese language charts . 
<|endoftext|> = = Plot = = 
<|endoftext|> Tōgō : The Story of Tōgō : The Story of Tōgō is a story of the same name . In this story , the demon Zayō ( 杭 ; 毉绘 , " Oan " Tōgō ) is revealed to be from the series 's epigod , having been converted into a human form by the demon Zayō . The story follows Tōgō and the three other demons , a reincarnation of his tribe Tōgō , and a clan leader named Nōgō . The four men are identified as Nōgō 's clan and , in the story , are featured in the story . The story is set in the summer of the monthly evening . 
<|endoftext|> The story of Tōgō is first presented in a different way ; the third installment was written by Zayō . The story is the only one in the story — the first one

Record

train loss
validation loss
tokens/sec
GPU utilization
VRAM usage
wall-clock training time

Option 3: Optimize nanoGPT

This is a natural next step if you're interested in systems work.

Implement and measure one optimization at a time:

FlashAttention variants
Fused optimizers
Gradient checkpointing
torch.compile tuning
Different attention implementations
Activation checkpointing
Mixed precision variations

Each optimization should be compared against your TinyStories baseline.


This is actually one of the best projects you can do on your DGX Spark. You're no longer asking "how do I train GPT-2?"—you're asking **"why is one implementation of GPT faster than another?"** That's systems engineering, and it's exactly how projects like nanoGPT, vLLM, TensorRT-LLM, and Megatron-LM evolve.

I would treat this as a **controlled benchmark suite**, where **each optimization is a separate experiment**. Never stack multiple changes until you've measured each one independently.

---

# Overall project structure

I would organize your repository like this:

```text
nanoGPT/
├── baselines/
│   ├── baseline.md
│   ├── flash_sdpa.md
│   ├── flash_v2.md
│   ├── xformers.md
│   ├── checkpointing.md
│   ├── compile.md
│   ├── fused_optimizer.md
│   └── ...
│
├── configs/
├── experiments/
├── results/
│   ├── baseline/
│   ├── flash_sdpa/
│   ├── flash_v2/
│   ├── ...
│
└── scripts/
```

Every experiment should have

* config
* logs
* checkpoint
* throughput
* GPU utilization
* conclusion

---

# Phase 0 — Establish a baseline

Before changing anything, record:

| Metric           | Value |
| ---------------- | ----- |
| tokens/sec       |       |
| iteration time   |       |
| MFU              |       |
| GPU utilization  |       |
| VRAM             |       |
| validation loss  |       |
| final train loss |       |
| power            |       |
| compile time     |       |

This baseline is your reference.

---

# Phase 1 — Understand FlashAttention

Many people think FlashAttention is "a faster attention kernel."

It isn't.

It is an entirely different algorithm.

Normal attention computes

```
QKᵀ

↓

Softmax

↓

SoftmaxV
```

while materializing the enormous

```
T × T
```

attention matrix.

FlashAttention never stores that matrix.

Instead it streams blocks through shared memory.

Advantages

* lower memory
* fewer HBM accesses
* higher occupancy
* much better for long contexts

---

# Phase 2 — Determine your current implementation

First determine what you're actually running.

If you're using modern nanoGPT, you'll likely find something like

```python
F.scaled_dot_product_attention(...)
```

PyTorch automatically dispatches to

* math kernel
* memory efficient kernel
* FlashAttention

depending on

* GPU
* dtype
* sequence length

So your baseline may already be using FlashAttention.

This is extremely important to determine first.

---

# Experiment A

## Disable FlashAttention

Force PyTorch to use the math kernel.

Measure

* tokens/sec
* memory
* MFU

This gives

```
Math Attention
```

---

# Experiment B

Enable FlashAttention

Now force

```
FlashAttention backend
```

Measure again.

Now you have

```
Math
↓

Flash
```

comparison.

---

# Experiment C

Memory Efficient Attention

PyTorch has another backend.

Compare

```
Math

Memory Efficient

Flash
```

---

# Experiment D

FlashAttention-2

Install

```
flash-attn
```

Replace attention module.

Measure

* throughput
* compile time
* memory

---

# Experiment E

Different sequence lengths

Repeat

```
128
256
512
1024
2048
```

You'll notice

FlashAttention becomes increasingly advantageous as sequence length grows.

---

# Phase 3 — Instrumentation

For every experiment collect

```
iteration

loss

tokens/sec

MFU

VRAM

GPU utilization

step time
```

Also

```
torch.profiler

nsys

ncu
```

if you want GPU kernel analysis.

---

# Phase 4 — Visualize

Create graphs.

Example

```
Context Length

128

256

512

1024

2048
```

vs

```
tokens/sec
```

Another

```
Memory
```

vs

```
Context Length
```

Another

```
MFU
```

vs

```
Attention Backend
```

---

# Phase 5 — Read the FlashAttention paper

While benchmarking, read

**FlashAttention (2022)**

Don't worry about every proof.

Focus on

* IO complexity
* tiled softmax
* SRAM vs HBM
* online softmax

Then

**FlashAttention-2**

Focus on

* better parallelization
* work partitioning
* occupancy improvements

---

# Suggested implementation order

I'd implement the attention variants in this order:

1. **PyTorch SDPA (baseline)** — confirm exactly which backend is selected on your DGX Spark.
2. **Force the math backend** — establish a true "no FlashAttention" baseline.
3. **Force the FlashAttention backend** — measure the improvement relative to math.
4. **Test the memory-efficient backend** — compare all three PyTorch SDPA options.
5. **Integrate FlashAttention-2** — use the external `flash-attn` package and benchmark it against PyTorch's implementation.

Only after you've finished those would I move on to fused optimizers, gradient checkpointing, and compile tuning.

---

# Why this makes an excellent systems project

By the end of these experiments, you won't just know that "FlashAttention is faster." You'll be able to answer questions like:

* How much faster is it on **your DGX Spark**?
* At what sequence length does it become worthwhile?
* How much VRAM does it save?
* Does it change numerical stability or convergence?
* How does PyTorch's built-in FlashAttention compare with FlashAttention-2?

Those are the kinds of measurements that make for a strong engineering report or technical blog post, because they're grounded in reproducible experiments rather than anecdotal claims.

This also sets up the rest of your optimization work: once you have a well-instrumented baseline and a repeatable benchmarking methodology, you can evaluate every subsequent optimization (fused optimizers, checkpointing, `torch.compile`, etc.) with the same rigor.
