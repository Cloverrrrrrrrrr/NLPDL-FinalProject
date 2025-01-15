Q_PROMPT = """You are a help assistant. Please solve the following math problem. Here are some examples, you should follow the same format to answer the question:
Question: We flip a fair coin 10 times.  What is the probability that we get heads in at least 6 of the 10 flips?
Answer:
Let's think step by step.
Step 1: There are $2^{10} = 1024$ possible outcomes of the 10 coin flips.
Step 2: The probability that we flip at least 6 heads is equal to the probability that we flip at least 6 tails, by symmetry.
Step 3: Let's call this probability $p$. The only other possibility is that we flip exactly 5 heads and 5 tails, for which the probability is $\\dfrac{\\binom{10}{5}}{2^{10}} = \\dfrac{252}{1024} = \\dfrac{63}{256}$. 
Step 4: Therefore, $\\dfrac{63}{256} + 2p = 1$, giving $$ p=\\frac{1}{2}\\left(1-\\frac{63}{256}\\right)=\\boxed{\\frac{193}{512}} $$

Question: For a particular peculiar pair of dice, the probabilities of rolling 1, 2, 3, 4, 5, and 6 on each die are in the ratio $1:2:3:4:5:6$. What is the probability of rolling a total of 7 on the two dice?
Answer:
Let's think step by step.
Step 1: On each die the probability of rolling $k$, for $1\\leq\nk\\leq 6$, is \\[\n\\frac{k}{1+2+3+4+5+6}=\\frac{k}{21}.\n\\]
Step 2: There are six ways of rolling a total of 7 on the two dice, represented by the ordered pairs $(1,6)$, $(2,5)$, $(3,4)$, $(4,3)$, $(5,2)$, and $(6,1)$.
Step 3: Thus the probability of rolling a total of 7 is \\[\n\\frac{1\\cdot6+2\\cdot5+3\\cdot4+4\\cdot3+5\\cdot2+6\\cdot1}{21^2}=\\frac{56}{21^2}=\\boxed{\\frac{8}{63}}.\n\\]

Question: At a school, all 60 students play on at least one of three teams: Basketball, Soccer, and Mathletics. 8 students play all three sports, half the students play basketball, and the ratio of the size of the math team to the size of the basketball team to the size of the soccer team is $4:3:2$. How many students at the school play on exactly two teams?
Answer:
Let's think step by step.
Step 1: We have enough information to solve for the size of each team. There are $\\dfrac{60}{2}=30$ members of the basketball team, $\\dfrac{4}{3}(30)=40$ members of the math team, and $\\dfrac{2}{3}(30)=20$ members of the soccer team.
Step 2: Adding these up gives us 90, so clearly we're overcounting since there are only 60 students.
Step 3: The number of times that each student is counted in this total is equal to the number of teams that student plays on. This means that all 60 students will be counted at least once, all students who play exactly two sports will be counted one extra time, and all students who play three sports will be counted two extra times.
Step 4: Letting $x$ be the number of students who play two sports and $y$ be the number who play all three gives us $60+x+2y=90$.
Step 5: However, we know that $y=8$, so we can substitute that in and get $x=\\boxed{14}$.

Question: A 12-slice pizza was made with only pepperoni and mushroom toppings, and every slice has at least one topping. Only five slices have pepperoni, and exactly eight slices have mushrooms. How many slices have both pepperoni and mushrooms?
Answer:
Let's think step by step.
Step 1: There are 12 total slices, and 5 of them have pepperoni and 8 have mushrooms. Let $n$ be the number that have both toppings. So, there are $5-n$ slices with only pepperoni and $8-n$ with only mushrooms.
Step 2:  So, there are a total of $n + (5-n) + (8-n)$ slices.  Since there are 12 slices, we have $(5-n) + (8-n) + n = 12$, from which we find that $n=\\boxed{1}$."""

JUDGE_PROMPT = "Please judge the correctness of the prediction based on the groundtruth answer. Your answer should be either 'true' or 'false'.\nGroundtruth: {}\nPrediction: {}\n"

CORRECT_PROMPT = """For the following math word problems, the original solutions may contain errors. Please identify the incorrect step in each solution, explain why it is incorrect, and provide the correct solution starting from that step.

Question: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?
Original Solution:
Step 1: Alexis spent a total of $30 + $46 + $38 + $11 + $18 = $143 on the clothes she has receipts for.
Step 2: She had a budget of $200, so she spent $200 - $143 = $57 on the shoes.
Step 3: The answer is $57.
Incorrect Step: Step 2.
Explanation: The problem states that Alexis has $16 left from her budget, not that she spent her entire budget. So, to find the amount she spent on the shoes, we need to subtract the amount she has left from the total amount she spent on the other clothes.
Correct Solution:
...
Step 2: Alexis has $16 left from her budget, so she spent $200 - $16 = $184 on all the clothes, including the shoes.
Step 3: To find the cost of the shoes, subtract the cost of the other clothes from the total amount spent: $184 - $143 = $41.
Step 4: The answer is 41.

Question: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?
Original Solution:
Step 1: Let x be the number of people on the ship the monster ate in the first hundred years.
Step 2: Then the number of people on the second ship is 2x, the number of people on the third ship is 4x, and the number of people on the fourth ship is 8x.
Step 3: The total number of people eaten by the monster is x + 2x + 4x + 8x = 15x.
Step 4: We know that the monster ate 847 people over three hundred years, so we can set up the equation 15x = 847.
Step 5: Solving for x, we get x = 56.
Step 6: Therefore, there were 56 people on the ship the monster ate in the first hundred years.
Step 7: The answer is 56.
Incorrect Step: Step 2.
Explanation: The original solution incorrectly assumes that there are four ships instead of three. Since the monster rises once every hundred years and we are considering a period of three hundred years, there should only be three ships.
Correct Solution:
...
Step 2: Then the number of people on the second ship is 2x, and the number of people on the third ship is 4x.
Step 3: The total number of people eaten by the monster is x + 2x + 4x = 7x.
Step 4: We know that the monster ate 847 people over three hundred years, so we can set up the equation 7x = 847.
Step 5: Solving for x, we get x = 121.
Step 6: Therefore, there were 121 people on the ship the monster ate in the first hundred years.
Step 7: The answer is 121.

Question: James creates a media empire. He creates a movie for $2000. Each DVD cost $6 to make. He sells it for 2.5 times that much. He sells 500 movies a day for 5 days a week. How much profit does he make in 20 weeks?
Original Solution:
Step 1: The total cost of creating the movie is $2000 + ($6 x 500) = $5000.
Step 2: He sells each DVD for 2.5 x $6 = $15.
Step 3: Thus, he earns $15 x 500 = $7500 per day.
Step 4: In a week, he earns $7500 x 5 = $37,500.
Step 5: In 20 weeks, he earns $37,500 x 20 = $750,000.
Step 6: His profit is $750,000 - $5000 = $745,000.
Step 7: The answer is 745000.
Incorrect Step: Step 1.
Explanation: The incorrect part of this step is the calculation of the total cost of creating the movie. The $2000 is a one-time cost for creating the movie, while the $6 cost per DVD should be multiplied by the total number of DVDs sold in 20 weeks, not just 500.
Correct Solution:
Step 1: Calculate the total number of DVDs sold in 20 weeks: 500 DVDs/day * 5 days/week * 20 weeks = 50,000 DVDs.
Step 2: Calculate the total cost of creating the movie and producing the DVDs: $2000 (movie creation) + ($6/DVD * 50,000 DVDs) = $2000 + $300,000 = $302,000.
Step 3: Calculate the selling price of each DVD: 2.5 * $6 = $15.
Step 4: Calculate the total revenue from selling the DVDs: $15/DVD * 50,000 DVDs = $750,000.
Step 5: Calculate the profit: $750,000 (revenue) - $302,000 (cost) = $448,000.
Step 6: The answer is 448000.

Question: It takes Roque two hours to walk to work and one hour to ride his bike to work. Roque walks to and from work three times a week and rides his bike to and from work twice a week. How many hours in total does he take to get to and from work a week with walking and biking?
Original Solution:
Step 1: Roque walks to work and back three times a week, so he spends 2 hours walking to work and 2 hours walking back from work for a total of 2 + 2 = 4 hours of walking per day.
Step 2: He rides his bike to work and back twice a week, so he spends 1 hour riding to work and 1 hour riding back from work for a total of 1 + 1 = 2 hours of biking per day.
Step 3: Therefore, Roque spends a total of 4 hours of walking + 2 hours of biking = 6 hours to get to and from work a week.
Step 4: The answer is 6.
Incorrect Step: Step 3.
Explanation: The incorrect step is Step 3 because it does not account for the number of days Roque walks and bikes to work. The total hours should be calculated by multiplying the hours spent walking and biking per day by the number of days he walks and bikes.
Correct Solution:
...
Step 3: Roque spends a total of (4 hours of walking per day * 3 days) + (2 hours of biking per day * 2 days) = 12 hours of walking + 4 hours of biking = 16 hours to get to and from work a week.
Step 4: The answer is 16.

Now correct the original solution based on the given groundtruth answer following the above format of the examples.
Question: {}
Groundtruth: {}
Original Solution:\n {}
"""