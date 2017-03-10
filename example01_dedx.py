
import dedx



dedxObj = dedx.energyLoss(85)
dedxObj.track(92, 238., 10., 1.e-4)

print dedxObj.eloss(), dedxObj.sigmaE(), dedxObj.sigmaAng(), dedxObj.elossAMEV(), dedxObj.sigmaEAMEV()
