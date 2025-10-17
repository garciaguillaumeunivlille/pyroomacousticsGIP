Generated-IRs/17-10-Brouillon

Test 1: 
    Profil : Brouillon
    S/M : F 1à16
    Mat : pra.Material(energy_absorption=0.01, scattering=0.3) 
    Result : ValueError: zero-size array to reduction operation maximum which has no identity
    Conclusion : Lancer la simulation avec tous les micro n'a pas abouti
    Durée: ~10/15min

        Test 1 Bis :
            Test en lançant les micros 1 par 1
            Durées individuelles : 27s
            Result :
                - Generation des IR ok
                - Une erreur "zero-size array"de temps en temps mais non persistente
            Observations :
                - F7 a éventuellement été générée mais a planté beaucoup plus que les autres