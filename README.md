# JOJO - Begrüßungs- und Interaktionsroboter

Das Ziel des Projekts ist die Entwicklung eines Roboters **JOJO**, der Menschen begrüßt und mit ihnen interagieren kann. Der Roboter ist in der Lage, eine normale menschliche Konversation zu führen und Personen anhand ihrer Gesichter zu erkennen und zu begrüßen. Das System wurde mit minimalem Budget und optimierter Leistung entwickelt.

---

## Technische Grundlage

- **Zentrale Steuerung:**
  - Ein **Raspberry Pi 4** steuert den gesamten Roboter und kommuniziert über WLAN mit der **Gemini AI**.
  - Zudem läuft ein **Gesichtserkennungs-Service** auf dem Raspberry Pi.
  - Der Pi wird durch eine **Powerbank** mit Energie versorgt.
  
- **Bewegungsplattform:**
  - Der Roboter verfügt über einen **Fahrkörper**, der mit **Lithium-Batterien** betrieben wird.
  - Die Steuerung erfolgt über einen **Arduino UNO**.
  - Der Fahrkörper wurde vom Professor **Kühebacher** anhand eines bestehenden Modells aus einer anderen Klasse inspiriert.
  
- **AI-Komponenten:**
  - **Sprachsteuerung:** Der Roboter kommuniziert mit der **Gemini AI**.
  - **Gesichtserkennung:** Implementierung über das GitHub-Repository [face_recognition](https://github.com/ageitgey/face_recognition).

---

## Konstruktion

- **Unterer Bereich:**
  - Fahrkörper
  - Lithium-Batterie
  - Powerbank
  - Raspberry Pi 4
  - Audio-Ausgabe-Box

- **Mittlerer Bereich:**
  - 1-Meter-Stange als tragende Struktur
  - Mikrofon (auf halber Höhe angebracht)

- **Oberer Bereich:**
  - 4 Kameras für eine **360-Grad-Sicht**

---

## Bewegungsmechanik

- **Fahrweise:** Der Roboter bewegt sich **zufällig** durch den Raum.
- **Hindernisvermeidung:**
  - Ein **Schallsensor** ist vorne am Roboter montiert und erkennt Hindernisse.
  - Erkennt der Sensor ein Hindernis, dreht sich der Roboter um **70 Grad** und setzt die Fahrt in eine andere Richtung fort.

---

## Sprach- und Gesichtserkennung

- **Aktivierung:** Der Sprachbefehl **"Jojo"** aktiviert die Sprachsteuerung.
- **Interaktion:** Der Roboter kommuniziert mithilfe eines **Gemini-AI-API-Keys**.
- **Gesichtserkennung:** Die AI erkennt **bekannte Gesichter** und kann sie benennen, basierend auf bereits gespeicherten Fotos der Person.
