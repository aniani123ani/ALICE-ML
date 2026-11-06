#include <iostream>
#include <vector>
#include <cmath>

#include <TFile.h>
#include <TTree.h>

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

#include "RootEvent.h"

using namespace std;
using namespace fastjet;

int main() {

    // the hard pp event file
    TFile *hardFile = TFile::Open("pp_hard.root", "READ");

    if (!hardFile || hardFile->IsZombie()) {
        cerr << "Could not open pp_hard.root" << endl;
        return 1;
    }

    // the event tree from the hard pp file
    TTree *hardTree = nullptr;
    hardFile->GetObject("t", hardTree);

    if (!hardTree) {
        cerr << "Could not find TTree t in pp_hard.root" << endl;
        return 1;
    }

    cout << "Hard events: " << hardTree->GetEntries() << endl;

    // the soft-QCD event file
    TFile *softFile = TFile::Open("pp_soft.root", "READ");

    if (!softFile || softFile->IsZombie()) {
        cerr << "Could not open pp_soft.root" << endl;
        return 1;
    }
// the event tree from the soft-QCD file
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


TFile *outputFile = TFile::Open("ml_dataset.root", "RECREATE");


TTree *outputTree = new TTree("t", "ml training dataset");

// the truth jet pT as a BDT input
double pt_jet;
outputTree->Branch("pt_jet", &pt_jet);

double pt_leading;
outputTree->Branch("pt_leading", &pt_leading);

double pt_soft_sum;
outputTree->Branch("pt_soft_sum", &pt_soft_sum);

double ppTruth;
outputTree->Branch("ppTruth", &ppTruth);

double PbPbtruth;
outputTree->Branch("PbPbtruth", &PbPbtruth);


int N = 20;
double R=0.4;
Long64_t NsoftEntries = softTree->GetEntries();
Long64_t Nsoft = -1;

for (Long64_t i = 0; i < nEvents; ++i) {

    
    hardTree->GetEntry(i);

    
    vector<PseudoJet> hardParticles;
    for (const auto &track : hardEvent->tracks) {


        double px = track.pT * cos(track.phi);
        double py = track.pT * sin(track.phi);
        double pz = track.pT * sinh(track.eta);
        double E  = track.pT * cosh(track.eta);

    
        hardParticles.push_back(PseudoJet(px, py, pz, E));
    }

JetDefinition jetDefinition(antikt_algorithm, R);

ClusterSequence hardCluster(hardParticles, jetDefinition);

vector<PseudoJet> hardJets = sorted_by_pt(hardCluster.inclusive_jets());
 



if (hardJets.empty()) continue;


PseudoJet ppJet = hardJets[0];

//keeping only truth jets above the 30 GeV threshold
if (ppJet.pt() < 30.0) continue;
if(abs(ppJet.eta()) >= 2.0) continue;
ppTruth = ppJet.pt();
pt_jet= ppJet.pt();
pt_leading = 0.0;

for (const auto &constituent : ppJet.constituents()) {

    
    if (constituent.pt() > pt_leading) {
        pt_leading = constituent.pt();
    }


}

/////////////////////////////////////////////////////////////////////
pt_soft_sum = 0.0;


vector<PseudoJet> combinedParticles = hardParticles;

// Add the first N soft-QCD events
for (int j = 0; j < N; ++j) {
    Nsoft++;
    if (Nsoft >= NsoftEntries) {
        Nsoft = 0; // Reset to the first soft event if we reach the end
    }

    softTree->GetEntry(Nsoft);

    for (const auto &track : softEvent->tracks) {
        double px = track.pT * cos(track.phi);
        double py = track.pT * sin(track.phi);
        double pz = track.pT * sinh(track.eta);
        double E  = track.pT * cosh(track.eta);

        PseudoJet softParticle(px, py, pz, E);

        // Calculating distance from the original hard-jet axis
        double deltaEta = track.eta - ppJet.eta();
        double deltaPhi = ppJet.delta_phi_to(softParticle);

        double deltaR = sqrt(
            deltaEta * deltaEta + deltaPhi * deltaPhi
        );

        // Suming soft-particle pT inside the R = 0.4 cone
        if (deltaR < R) {
            pt_soft_sum += track.pT;
        }

        // Add every soft particle to the combined event
        combinedParticles.push_back(softParticle);
    }
}

/////////////////////////////////////////////////////////////////
ClusterSequence combinedCluster(combinedParticles, jetDefinition);

// Get the resulting jets ordered by decreasing pT
vector<PseudoJet> combinedJets =
    sorted_by_pt(combinedCluster.inclusive_jets());

double minDeltaR = 999.0;
int matchedJetIndex = -1;

for (size_t k = 0; k < combinedJets.size(); ++k) {

    double deltaR = ppJet.delta_R(combinedJets[k]);

    if (deltaR < minDeltaR) {
        minDeltaR = deltaR;
        matchedJetIndex = k;
    }
}

if (matchedJetIndex < 0 || minDeltaR >= 0.2) continue;


// the pT of the matched jet after adding the soft background
if (matchedJetIndex >= 0) {
    PbPbtruth = combinedJets[matchedJetIndex].pt();
} else {
    continue;
}
outputTree ->Fill();
}

outputTree->Write();

outputFile->Close();

hardFile->Close();
softFile->Close();

return 0;
}

