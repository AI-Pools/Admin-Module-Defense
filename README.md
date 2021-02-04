 # Bienvenue dans le module *Time Out* du programe *Far From Earth*
```
     
                                           ..,,;;;;;;,,,,
                                     .,;'';;,..,;;;,,,,,.''';;,..
                                  ,,''                    '';;;;,;''
                                 ;'    ,;@@;'  ,@@;, @@, ';;;@@;,;';.
                                ''  ,;@@@@@'  ;@@@@; ''    ;;@@@@@;;;;
                                   ;;@@@@@;    '''     .,,;;;@@@@@@@;;;
                                  ;;@@@@@@;           , ';;;@@@@@@@@;;;.
                                   '';@@@@@,.  ,   .   ',;;;@@@@@@;;;;;;
                                      .   '';;;;;;;;;,;;;;@@@@@;;' ,.:;'
                                        ''..,,     ''''    '  .,;'
                                             ''''''::''''''''
                                                                 ,;
                                {Module Defense}                .;;
                                                               ,;;;
                                --Far From Earth--           ,;;;;:
                                                          ,;@@   .;
                                                         ;;@@'  ,;
                                                         ';;, ,;'        [04/02/2021]
```

--- 
 
 
Bonjour l'équipe technique ! Je suis heureux de vous voir toujours d'aplomb ! Vous venez de faire vos premiers pas dans le monde du Reinforcement Learning qui est un domaine de l'IA souvent utiliser dans la robotique et les jeux vidéo ! 
 
 
Le Reinforcement Learning est souvent découpé en deux types d'algorithmes : Value based et Policy Based. Vous venez de voir le Q Learning et le Deep Q Learning qui sont des algorithmes « Value Based ». Maintenant vous allez voir certains algorithmes « Policy Based ». 
 
 
Les deux types ont leurs avantages et leur inconvénient et dans certains algorithmes on mixe mêmes les deux pour en récupérer le meilleur et donc créer des algorithmes capables de résoudre des environnements extrêmement complexes ! 
 
 
Voilà pourquoi nous vous apprenons les deux types et pas un seul. 
Nous allons  
 
 
Grace a ces nouvelles compétences en plus vous allez pouvoir résoudre de nouvelles missions et agrandir votre bagage de connaissance en intelligence artificielle. 
 
 
Le première algorithme « Policy Based » que vous allez voir s’appel : Policy Gradient. A la différence du Deep Q Learning Policy Gradient est capable de gérer des actions continue. Autrement dit il peut par exemple tourner un volant de 20.5 degrés vers la droite. (je vous invite à chercher la différence entre action continue et action discrète).  
 
 
Policy gradient est également capable de gérer des actions discrètes. Et c’est ce que vous allez voir maintenant car il est plus facile à voir au début. 
 
 
Avant de rentré dans le vif du sujet nous allons parler d’une chose importante dans la création d'algorithmes et d’IA en général. Lorsque vous voulez résoudre un problème complexe il faut tout d’abord tester votre algorithme sur quelque chose de classique et de simple pour s'assurer que tout fonctionne comme il faut sur quelque chose de basique.  
 
 
Une fois que vous avez la certification que votre algorithme fonctionne sur quelque chose de basique vous pouvez commencer à le tester sur quelque chose de plus compliqué. 
 
 
Vous avez vu dernièrement l'environnement GYM « Cartpole » et c’est un environnement assez basique qui va justement vous permettre de tester vos IA de Reinforcement Learning. 
 
 
Vous devez donc résoudre l'environnement GYM « Cartpole » grâce à la méthode de Reinforcement Learning Policy Gradient. 
 
 
Voici quelques directives :  
 
 
- Vous devez avoir un réseau de neurone avec des couche dense et linéar qui vous ressortira la meilleure action à prendre parmi la liste d'action possible. (Regardez comment fonctionne un environnement gym et comment nous pouvons récupérer le nombre d'action possible et la shape des inputs que l'on nous passe) 
- Vous devez également avoir une classe Memory comme dans les exercices de Reinforcement Learning précèdent pour stocker des Transitions. La classe Memory doit pouvoir stocker et récupérer les Transitions, ainsi que clear l'entièreté de la mémoire. 
- Vous devez ensuite créer une classe Agent qui va représenter notre agent : 
- Un agent possède une mémoire et doit pouvoir prendre des actions. Il faut donc lui donner une instance de la classe mémoire et de la classe réseau de neurone. 
- Pour prendre des actions cette agent doit implémenter une fonction `take_action()` qui va utiliser la technique de Policy Gradient pour prendre une action avec le réseau de neurones qu'il possède et en prenant son état actuel comme paramètre du réseau de neurones. 
- Si vous avez bien compris la technique de Policy gradient vous avez vu que l'on calcule ce qu'on appelle les Qval à partir de toute les récompense de l'épisode. Donc nous devons faire une fonction qui prend en paramètre les reward de l'épisode actuel puis qui calcule les Qvals de chaque transition en partant de la fin des rewards de l'épisode. (Un épisode ici est le fait de commencer le jeu et de gagner ou de perdre. On recommence cela plusieurs fois. Un épisode est composé de plusieurs transitions)  
### Maintenant que vous avez toutes les fonctions utilitaires il faut maintenant les assembler pour créer la fonction d'apprentissage `learn()` : 
- Cette fonction récupère toutes les transitions de l'épisode soit ` n * ('state', 'action', 'next_state', 'reward')` mais sous forme de 4 tableaux diffèrent. Ensuite elle calcule les qvals de chaque transitions grâce à la fonction que vous avez créé précédemment en lui passant le tableau de reward de l'episode. Une fois que nous avons tout cela on applique la technique de policy gradient.  
> Tips : 
> - je n'oublie pas de reset les gradients avec mon optimizer. 
> - je n'oublie pas d'utiliser un softmax pour avoir une répartition de probabilité (Rechercher a quoi sert un softmax puis appeler un encadrant). 
> - Je calcule la loss (la suite de la technique policy gradient) 
> - J'utilise la fonction` backward()` sur la loss pour calculer les gradients puis la fonction `step()` pour appliquer les changements. Suite à cela je `clear()` la mémoire pour dire que l'épisode est terminé et que l'on passe au suivant. 
### Il faut maintenant créer un `main()` pour créer un agent et lui passer tout ce dont il a besoin pour apprendre puis prendre des décisions : 
- Tout d'abord on récupère l'environnement GYM en question. 
- Je créer une instance de mon réseau de neurone, je lui passe la shape d'entré et de sortie de l'environnement. 
- Je créer un optimizer. 
- je créer un agent et je lui passe l'optimizer et le réseau de neurone 
- Maintenant il faut faire une double boucle pour itérer sur le nombre d'épisodes que je veux faire puis sur le nombre d'étape (transitions) maximum qu'un épisode peut faire (sachant que si gym nous renvoie 'DONE' cela veut dire que l'épisode est terminé). 
- Si l'épisode est terminé, je dois appeler ma fonction `learn()` de mon agent. Dans le cas contraire à chaque étape de l'épisode je récupère l'état de mon environnement je le passe dans la fonction `take_action()` de mon agent. Celui-ci me retourne l'action à prendre et je passe cette action dans la fonction `step()` de mon environnement gym qui me permet de récupérer `new_state, reward, done` . 
Je stock tout cela dans ma mémoire pour que quand l'épisode se termine mon agent récupère ces données puis les utilises pour apprendre 
Je n'oublie pas de clear la memory pour le prochain épisode. 
 
 
Tentez maintenant de faire apprendre votre agent *Cartpole*. Il doit être capable de faire tenir le bâton en l'air pendant 499 itérations/Transitions (ce qui est le maximum). 

Vous venez de crée votre premier algorithme de Deep Reinforcement Learning (Réseau de neurone + Reinforcement Learning). Dans la prochaine missions vous allez reprendre les notions de ce module pour crée un Module de defense advanced qui va s'occuper de tirer sur les vaiseau extraterestre qui voudrait entré dans l'atmostphere de la terre.

---
# A suivre ...

Vous venez de crée votre premier algorithme de Deep Reinforcement Learning (Réseau de neurone + Reinforcement Learning). Dans la prochaine mission vous allez reprendre les notions de ce module pour crée un Module de défense Advanced qui va s'occuper de tirer sur les vaisseaux extraterrestres qui voudrait entrer dans l'atmosphère de la terre. Il faudra donc un algo précis comme vous l'avez fait ici mais nous allons lui ajouter un petit boost. 