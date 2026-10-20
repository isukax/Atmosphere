#include "atmosphere/PhaseMoments.h"

#include "atmosphere/AtmosphereParams.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace atmosphere {

PhaseMoments legendreMomentsTable(const std::vector<glm::vec4>& table)
{
    if (table.size() != kMiePhaseTableCount) {
        throw std::runtime_error("legendreMomentsTable: table must have " + std::to_string(kMiePhaseTableCount) +
                                 " points");
    }
    // [POINT] u で積分する: μ = cos(π·u²)、dμ = −sin(π·u²)·2π·u du。表は u の等間隔で、点の間は u で線形
    //         （MiePhase と同じ補間）。前方のピーク（u が小さい所）も u では滑らかなので、区間ごとに 64 点の中点則で
    //         十分（区間数 2,048 × 64、倍精度）
    constexpr double kPi = std::numbers::pi;
    constexpr int kSub = 64;
    const int intervals = static_cast<int>(table.size()) - 1;
    const double du = 1.0 / (static_cast<double>(intervals) * kSub);
    double sum[3][3] = {};
    for (int i = 0; i < intervals; ++i) {
        for (int k = 0; k < kSub; ++k) {
            const double f = (k + 0.5) / kSub;
            const double u = (i + f) * (1.0 / intervals);
            const double mu = std::cos(kPi * u * u);
            const double weight = 2.0 * kPi * std::sin(kPi * u * u) * 2.0 * kPi * u * du;
            const double legendre[3] = {1.0, mu, 0.5 * (3.0 * mu * mu - 1.0)};
            for (int c = 0; c < 3; ++c) {
                const double p = (1.0 - f) * table[i][c] + f * table[i + 1][c];
                for (int l = 0; l < 3; ++l) {
                    sum[l][c] += p * legendre[l] * weight;
                }
            }
        }
    }
    PhaseMoments chi;
    for (int l = 0; l < 3; ++l) {
        chi[l] = glm::vec3(static_cast<float>(sum[l][0]), static_cast<float>(sum[l][1]), static_cast<float>(sum[l][2]));
    }
    return chi;
}

PhaseMoments legendreMomentsCornetteShanks(float gf)
{
    // [POINT] Cornette-Shanks = 3/(2(2+g²))·(1+μ²)·HG(μ)。HG の χ_l = g^l と μ²·P_l の Legendre 展開から
    //           χ_1 = 3g(4+g²) / (5(2+g²))
    //           χ_2 = 3/(2(2+g²)) · (2/15 + (32/21)g² + (12/35)g⁴)
    //         （g = 0 で Rayleigh の (1, 0, 0.1)、g = 0.8 で χ_1 = 0.8436）
    const double g = gf;
    const double g2 = g * g;
    const double chi1 = 3.0 * g * (4.0 + g2) / (5.0 * (2.0 + g2));
    const double chi2 = 3.0 / (2.0 * (2.0 + g2)) * (2.0 / 15.0 + 32.0 / 21.0 * g2 + 12.0 / 35.0 * g2 * g2);
    return {glm::vec3(1.0f), glm::vec3(static_cast<float>(chi1)), glm::vec3(static_cast<float>(chi2))};
}

MiePhaseMoments mieLegendreMoments(const AtmosphereParams& p)
{
    MiePhaseMoments m;
    m.component[0] = p.miePhaseTableSrv == kNoMiePhaseTable ? legendreMomentsCornetteShanks(p.miePhaseG)
                                                            : p.miePhaseMoments;
    if (p.mieComponentCount > 1) {
        m.component[1] = p.mie1.phaseTableSrv == kNoMiePhaseTable ? legendreMomentsCornetteShanks(p.mie1.phaseG)
                                                                  : p.mie1.phaseMoments;
    }
    return m;
}

} // namespace atmosphere
