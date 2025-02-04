# ECE Cup - Pi-Car Autonomous Robot

## 🏆 Projet ECE Cup - Automatisation d'un Robot Pi-Car

Ce projet a été réalisé dans le cadre de l'ECE Cup. L'objectif était d'automatiser un robot Pi-Car afin qu'il puisse naviguer de manière autonome en combinant plusieurs fonctionnalités : suivi de ligne, reconnaissance de formes et de couleurs via OpenCV, et détection d'obstacles grâce aux capteurs à ultrasons.

---

## 🚀 Fonctionnalités

### 🔹 Suivi de ligne
- Utilisation d'une caméra pour détecter et suivre une ligne tracée sur le sol.
- Traitement d'images avec OpenCV pour extraire les contours et ajuster la trajectoire du robot.

### 🎨 Reconnaissance de formes et de couleurs
- Détection de formes géométriques et de couleurs spécifiques via OpenCV.
- Prise de décisions basées sur les objets détectés (exemple : changement de direction en fonction des couleurs ou formes rencontrées).

### 📡 Communication et Exécution des Scripts
- Connexion en SSH au Raspberry Pi du Pi-Car pour exécuter les scripts de contrôle.
- Interface en Python permettant d'envoyer des commandes au robot en temps réel.

### 🏁 Labyrinthe et évitement d'obstacles
- Intégration de capteurs à ultrasons pour détecter les obstacles et ajuster la trajectoire.
- Algorithme de navigation pour éviter les collisions et se déplacer efficacement à travers un labyrinthe.

---

## 🔧 Technologies et Matériel Utilisés

- **Matériel :** Pi-Car (Robot sur base Raspberry Pi), caméra embarquée, capteurs à ultrasons.
- **Logiciels & Frameworks :**
  - OpenCV (Traitement d'images)
  - Python (Contrôle et traitement des données)
  - SSH (Communication et exécution des scripts)
  - NumPy et SciPy (Traitement mathématique)

---

## 🛠️ Installation et Exécution

### 📥 Prérequis
- Un Raspberry Pi avec Raspbian installé
- Python 3 et les bibliothèques suivantes :
  ```bash
  pip install opencv-python numpy scipy paramiko
  ```
- Connexion SSH configurée pour exécuter des commandes à distance

### 🚀 Exécution
1. **Connexion SSH** :
   ```bash
   ssh pi@adresse_ip_du_robot
   ```
2. **Lancer le script principal** :
   ```bash
   python main.py
   ```
3. Observer le robot analyser son environnement et se déplacer intelligemment 🚗💨

---

## 📌 Améliorations Possibles
- Optimisation des algorithmes de traitement d'image pour une reconnaissance plus rapide.
- Ajout d'une interface graphique pour faciliter le contrôle du robot.
- Implémentation d'une intelligence artificielle pour améliorer la navigation dans le labyrinthe.

---

## 👥 Auteurs
- **Shaima Derouich** & Team 🎯

Si tu as des questions ou suggestions, n'hésite pas à contribuer ou à me contacter ! 🚀
