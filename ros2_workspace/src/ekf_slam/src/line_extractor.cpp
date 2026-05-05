#include "ekf_slam/line_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ekf_slam {

LineExtractor::LineExtractor() : LineExtractor(Params{}) {}

LineExtractor::LineExtractor(const Params& p) : params_(p) {
    // Cap per-bucket size at ~all-beams * buffer length so a long-lived bucket
    // doesn't grow without bound.
    bucket_cap_ = std::max<size_t>(40, params_.buffer_scans * 4);
}

double LineExtractor::normalizeAngle(double a) {
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

int LineExtractor::bucketOfWorldBearing(double bearing_rad, int bucket_deg) {
    double deg = bearing_rad * 180.0 / M_PI;
    while (deg >   180.0) deg -= 360.0;
    while (deg <= -180.0) deg += 360.0;
    int rounded = static_cast<int>(std::lround(deg / bucket_deg)) * bucket_deg;
    if (rounded >  180) rounded -= 360;
    if (rounded <= -180) rounded += 360;
    return rounded;
}

void LineExtractor::addScan(double x, double y, double yaw,
                            const std::array<double, 4>& ranges_b,
                            const std::array<double, 4>& bearings_b) {
    seq_ += 1.0;

    for (size_t i = 0; i < 4; ++i) {
        double r = ranges_b[i];
        if (!std::isfinite(r)) continue;
        if (r < params_.min_range || r >= params_.max_range) continue;

        double world_bearing = normalizeAngle(yaw + bearings_b[i]);
        int    bucket        = bucketOfWorldBearing(world_bearing, params_.bucket_deg);

        Endpoint ep;
        ep.wx    = x + r * std::cos(world_bearing);
        ep.wy    = y + r * std::sin(world_bearing);
        ep.t_seq = seq_;

        auto& dq = buckets_[bucket];
        dq.push_back(ep);
        while (dq.size() > bucket_cap_) dq.pop_front();
    }

    // Cull entries older than buffer_scans worth of scans so a bucket the drone
    // has moved past doesn't keep stale points indefinitely.
    double min_seq = seq_ - static_cast<double>(params_.buffer_scans);
    for (auto it = buckets_.begin(); it != buckets_.end();) {
        auto& dq = it->second;
        while (!dq.empty() && dq.front().t_seq < min_seq) dq.pop_front();
        if (dq.empty()) {
            it = buckets_.erase(it);
        } else {
            ++it;
        }
    }
}

bool LineExtractor::fitLineWorld(const std::vector<Eigen::Vector2d>& pts,
                                 double& rho_w, double& theta_w) {
    const size_t n = pts.size();
    if (n < 2) return false;

    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const auto& p : pts) mean += p;
    mean /= static_cast<double>(n);

    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    double xmin = +std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = +std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    for (const auto& p : pts) {
        double dx = p.x() - mean.x();
        double dy = p.y() - mean.y();
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
        xmin = std::min(xmin, p.x()); xmax = std::max(xmax, p.x());
        ymin = std::min(ymin, p.y()); ymax = std::max(ymax, p.y());
    }

    double x_range = xmax - xmin;
    double y_range = ymax - ymin;

    Eigen::Vector2d normal_w;
    if (x_range >= y_range) {
        // y = m*x + b
        if (sxx <= 0.0) return false;
        double m = sxy / sxx;
        Eigen::Vector2d nrm(-m, 1.0);
        nrm.normalize();
        normal_w = nrm;
    } else {
        // x = m*y + b
        if (syy <= 0.0) return false;
        double m = sxy / syy;
        Eigen::Vector2d nrm(1.0, -m);
        nrm.normalize();
        normal_w = nrm;
    }

    double rho = normal_w.dot(mean);
    if (rho < 0.0) {
        normal_w = -normal_w;
        rho      = -rho;
    }
    rho_w   = rho;
    theta_w = std::atan2(normal_w.y(), normal_w.x());
    return true;
}

void LineExtractor::worldLineToRobot(double rho_w, double theta_w,
                                     double rx, double ry, double ryaw,
                                     double& rho_r, double& theta_r) {
    // Hessian-form transform: line {p : n_w . p = rho_w} viewed from robot
    // (rx, ry, ryaw):
    //   theta_r = wrap(theta_w - ryaw)
    //   rho_r   = rho_w - rx*cos(theta_w) - ry*sin(theta_w)
    theta_r = normalizeAngle(theta_w - ryaw);
    rho_r   = rho_w - rx * std::cos(theta_w) - ry * std::sin(theta_w);
    if (rho_r < 0.0) {
        rho_r   = -rho_r;
        theta_r = normalizeAngle(theta_r + M_PI);
    }
}

void LineExtractor::extract(double cur_x, double cur_y, double cur_yaw,
                            std::vector<LineObs>& out_lines,
                            std::vector<CornerObs>& out_corners) const {
    out_lines.clear();
    out_corners.clear();

    struct WorldFit { double rho_w; double theta_w; int n_points; };
    std::map<int, WorldFit> world_fits;

    for (const auto& kv : buckets_) {
        int bucket_deg = kv.first;
        const auto& dq = kv.second;
        if (dq.size() < params_.min_points) continue;

        // Sort endpoints along the perpendicular-to-bearing axis so gap-split
        // sees spatially adjacent points, not temporal order.
        double bucket_rad = bucket_deg * M_PI / 180.0;
        double ux = -std::sin(bucket_rad);
        double uy =  std::cos(bucket_rad);

        std::vector<std::pair<double, Eigen::Vector2d>> proj;
        proj.reserve(dq.size());
        for (const auto& ep : dq) {
            double s = ep.wx * ux + ep.wy * uy;
            proj.emplace_back(s, Eigen::Vector2d(ep.wx, ep.wy));
        }
        std::sort(proj.begin(), proj.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Gap-split: keep the longest contiguous segment.
        std::vector<Eigen::Vector2d> cur_seg;
        std::vector<Eigen::Vector2d> best_seg;
        double prev_s = std::numeric_limits<double>::quiet_NaN();
        for (const auto& pr : proj) {
            double s = pr.first;
            if (!cur_seg.empty() && (s - prev_s) > params_.gap_thresh) {
                if (cur_seg.size() > best_seg.size()) best_seg = cur_seg;
                cur_seg.clear();
            }
            cur_seg.push_back(pr.second);
            prev_s = s;
        }
        if (cur_seg.size() > best_seg.size()) best_seg = cur_seg;

        if (best_seg.size() < params_.min_points) continue;

        double rho_w, theta_w;
        if (!fitLineWorld(best_seg, rho_w, theta_w)) continue;

        world_fits[bucket_deg] = {rho_w, theta_w, static_cast<int>(best_seg.size())};

        LineObs lo;
        worldLineToRobot(rho_w, theta_w, cur_x, cur_y, cur_yaw, lo.rho, lo.theta);
        lo.cov         = Eigen::Matrix2d::Zero();
        lo.cov(0, 0)   = params_.line_var_rho;
        lo.cov(1, 1)   = params_.line_var_theta;
        lo.n_points    = static_cast<int>(best_seg.size());
        lo.bucket_deg  = bucket_deg;
        lo.rho_w       = rho_w;
        lo.theta_w     = theta_w;
        out_lines.push_back(lo);
    }

    if (world_fits.size() < 2) return;

    // Corners from pairs of nearly-perpendicular world lines.
    for (auto it_a = world_fits.begin(); it_a != world_fits.end(); ++it_a) {
        auto it_b = std::next(it_a);
        for (; it_b != world_fits.end(); ++it_b) {
            double t1 = it_a->second.theta_w;
            double t2 = it_b->second.theta_w;
            double r1 = it_a->second.rho_w;
            double r2 = it_b->second.rho_w;

            double dtheta = normalizeAngle(t1 - t2);
            if (std::abs(std::cos(dtheta)) >= params_.corner_perp_cos) continue;

            // Solve {[cos t1, sin t1],[cos t2, sin t2]} * [cx,cy]^T = [r1,r2]^T
            double a = std::cos(t1), b = std::sin(t1);
            double c = std::cos(t2), d = std::sin(t2);
            double det = a * d - b * c;
            if (std::abs(det) < 1e-6) continue;
            double cx_w = ( d * r1 - b * r2) / det;
            double cy_w = (-c * r1 + a * r2) / det;

            double dx = cx_w - cur_x;
            double dy = cy_w - cur_y;
            if (std::hypot(dx, dy) > params_.corner_max_dist) continue;

            double cs = std::cos(-cur_yaw), sn = std::sin(-cur_yaw);
            CornerObs co;
            co.x           = cs * dx - sn * dy;
            co.y           = sn * dx + cs * dy;
            co.cov         = Eigen::Matrix2d::Zero();
            co.cov(0, 0)   = params_.corner_var_xy;
            co.cov(1, 1)   = params_.corner_var_xy;
            co.bucket1_deg = it_a->first;
            co.bucket2_deg = it_b->first;
            co.xw          = cx_w;
            co.yw          = cy_w;
            out_corners.push_back(co);
        }
    }
}

}  // namespace ekf_slam
