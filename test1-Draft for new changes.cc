
#include <iostream>
#include <vector>
#include <cmath>

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

#include "RootEvent.h"

using namespace std;
using namespace fastjet;

struct SoftTrack {
    double pt;
    double eta;
    double phi;
};

// Calculate wrapped azimuthal-angle difference.
double deltaPhi(double phi1, double phi2) {
    return atan2(sin(phi1 - phi2), cos(phi1 - phi2));
}

int main() {

    // Hard pp event file
    TFile *hardFile = TFile::Open("pp_hard.root", "READ");

    if (!hardFile || hardFile->IsZombie()) {
        cerr << "Could not open pp_hard.root" << endl;
        return 1;
    }

    TTree *hardTree = nullptr;
    hardFile->GetObject("t", hardTree);

    if (!hardTree) {
        cerr << "Could not find TTree t in pp_hard.root" << endl;
        return 1;
    }

    cout << "Hard events: " << hardTree->GetEntries() << endl;

    // Soft-QCD event file
    TFile *softFile = TFile::Open("pp_soft.root", "READ");

    if (!softFile || softFile->IsZombie()) {
        cerr << "Could not open pp_soft.root" << endl;
        return 1;
    }

    TTree *softTree = nullptr;
    softFile->GetObject("t", softTree);

    if (!softTree) {
        cerr << "Could not find TTree t in pp_soft.root" << endl;
        return 1;
    }

    cout << "Soft events: " << softTree->GetEntries() << endl;

    RootEvent *hardEvent = nullptr;
    RootEvent *softEvent = nullptr;

    hardTree->SetBranchAddress("events", &hardEvent);
    softTree->SetBranchAddress("events", &softEvent);

    Long64_t nEvents = hardTree->GetEntries();
    Long64_t NsoftEntries = softTree->GetEntries();

    if (NsoftEntries == 0) {
        cerr << "No soft events available!" << endl;
        return 1;
    }

    // New output file: original dataset is preserved.
    TFile *outputFile =
        TFile::Open("bdt_dataset_randomcones.root", "RECREATE");

    TTree *outputTree =
        new TTree("t", "BDT dataset with two random cones");

    // BDT input features
    double pt_jet;
    double pt_leading;
    double avg_sums;

    // Truth information
    double ppTruth;
    double PbPbtruth;

    outputTree->Branch("pt_jet", &pt_jet);
    outputTree->Branch("pt_leading", &pt_leading);
    outputTree->Branch("avg_sums", &avg_sums);
    outputTree->Branch("ppTruth", &ppTruth);
    outputTree->Branch("PbPbtruth", &PbPbtruth);

    // Number of soft-QCD events mixed with each hard event
    const int N = 20;

    // Jet radius
    const double R = 0.4;

    // Random-cone radius
    const double randomConeR = 0.4;

    
    const double trackEtaMin = -2.0;
    const double trackEtaMax =  2.0;

    const double coneEtaMin = trackEtaMin + randomConeR;
    const double coneEtaMax = trackEtaMax - randomConeR;

    const double PI = acos(-1.0);

    TRandom3 randomNumber(0);

    Long64_t Nsoft = -1;

    // Main hard-event loop
    for (Long64_t i = 0; i < nEvents; ++i) {

        hardTree->GetEntry(i);

        vector<PseudoJet> hardParticles;

        // Reading hard particles
        for (const auto &track : hardEvent->tracks) {

            double px = track.pT * cos(track.phi);
            double py = track.pT * sin(track.phi);
            double pz = track.pT * sinh(track.eta);
            double E  = track.pT * cosh(track.eta);

            hardParticles.push_back(PseudoJet(px, py, pz, E));
        }

        // Reconstruct hard jets
        JetDefinition jetDefinition(antikt_algorithm, R);

        ClusterSequence hardCluster(hardParticles, jetDefinition);

        vector<PseudoJet> hardJets =
            sorted_by_pt(hardCluster.inclusive_jets());

        if (hardJets.empty()) continue;

        // Leading hard jet
        PseudoJet ppJet = hardJets[0];

        // Hard-jet selection
        if (ppJet.pt() < 30.0) continue;
        if (abs(ppJet.eta()) >= 2.0) continue;

        ppTruth = ppJet.pt();
        pt_jet = ppJet.pt();

        // Leading constituent pT
        pt_leading = 0.0;

        for (const auto &constituent : ppJet.constituents()) {
            if (constituent.pt() > pt_leading) {
                pt_leading = constituent.pt();
            }
        }

        
        vector<PseudoJet> combinedParticles = hardParticles;

        vector<SoftTrack> allSoftTracks;

        
        for (int j = 0; j < N; ++j) {

            Nsoft++;

            if (Nsoft >= NsoftEntries) {
                Nsoft = 0;
            }

            softTree->GetEntry(Nsoft);

            for (const auto &track : softEvent->tracks) {

                double px = track.pT * cos(track.phi);
                double py = track.pT * sin(track.phi);
                double pz = track.pT * sinh(track.eta);
                double E  = track.pT * cosh(track.eta);

                PseudoJet softParticle(px, py, pz, E);

                combinedParticles.push_back(softParticle);

                
                allSoftTracks.push_back({
                    track.pT,
                    track.eta,
                    track.phi
                });
            }
        }

        // -------------------------------------------------
        // Generate two random, non-overlapping cones
        // -------------------------------------------------

        double coneEta1, conePhi1;
        double coneEta2, conePhi2;

        bool validCones = false;

        while (!validCones) {

            coneEta1 = randomNumber.Uniform(
                coneEtaMin, coneEtaMax
            );

            conePhi1 = randomNumber.Uniform(0.0, 2.0 * PI);

            coneEta2 = randomNumber.Uniform(
                coneEtaMin, coneEtaMax
            );

            conePhi2 = randomNumber.Uniform(0.0, 2.0 * PI);

            double dEta = coneEta1 - coneEta2;

            double dPhi = deltaPhi(conePhi1, conePhi2);

            double separation = sqrt(
                dEta * dEta + dPhi * dPhi
            );

            
            if (separation >= 2.0 * randomConeR) {
                validCones = true;
            }
        }

        
        double sum1 = 0.0;
        double sum2 = 0.0;

        for (const auto &track : allSoftTracks) {

            double dEta1 = track.eta - coneEta1;
            double dPhi1 = deltaPhi(track.phi, conePhi1);

            double deltaR1 = sqrt(
                dEta1 * dEta1 + dPhi1 * dPhi1
            );

            if (deltaR1 < randomConeR) {
                sum1 += track.pt;
            }

            double dEta2 = track.eta - coneEta2;
            double dPhi2 = deltaPhi(track.phi, conePhi2);

            double deltaR2 = sqrt(
                dEta2 * dEta2 + dPhi2 * dPhi2
            );

            if (deltaR2 < randomConeR) {
                sum2 += track.pt;
            }
        }

        
        avg_sums = (sum1 + sum2) / 2.0;

        
        ClusterSequence combinedCluster(
            combinedParticles, jetDefinition
        );

        vector<PseudoJet> combinedJets =
            sorted_by_pt(combinedCluster.inclusive_jets());

        // Match combined jet to original hard jet
        double minDeltaR = 999.0;
        int matchedJetIndex = -1;

        for (size_t k = 0; k < combinedJets.size(); ++k) {

            double deltaR = ppJet.delta_R(combinedJets[k]);

            if (deltaR < minDeltaR) {
                minDeltaR = deltaR;
                matchedJetIndex = k;
            }
        }

        if (matchedJetIndex < 0 || minDeltaR >= 0.2) {
            continue;
        }

        PbPbtruth = combinedJets[matchedJetIndex].pt();

        outputTree->Fill();

        if (i % 1000 == 0) {
            cout << "Processed hard event " << i
                 << ", average random-cone sum = "
                 << avg_sums << " GeV" << endl;
        }
    }

    outputTree->Write();

    cout << "Dataset entries: " << outputTree->GetEntries()
         << endl;

    outputFile->Close();
    hardFile->Close();
    softFile->Close();

    return 0;
}

