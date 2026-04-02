#include "tone_curve_editor.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>

ToneCurveCanvas::ToneCurveCanvas(QWidget* parent) : QWidget(parent)
{
    setMinimumSize(320, 320);
    setMouseTracking(true);
    controlPoints_ = {
        QPointF(0.0, 0.0),
        QPointF(0.25, 0.18),
        QPointF(0.5, 0.5),
        QPointF(0.75, 0.82),
        QPointF(1.0, 1.0),
    };
}

void ToneCurveCanvas::setControlPoints(const std::vector<QPointF>& points)
{
    if (points.size() < 2) return;
    controlPoints_ = points;
    clampAndSortControlPoints();
    update();
}

void ToneCurveCanvas::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.fillRect(rect(), QColor(18, 18, 28));

    QRect graph = graphRect();
    p.fillRect(graph, QColor(10, 10, 16));
    p.setPen(QColor(50, 50, 70));
    p.drawRect(graph);

    for (int i = 1; i < 4; ++i) {
        int x = graph.left() + (graph.width() - 1) * i / 4;
        int y = graph.top() + (graph.height() - 1) * i / 4;
        p.drawLine(x, graph.top(), x, graph.bottom());
        p.drawLine(graph.left(), y, graph.right(), y);
    }

    QPainterPath path;
    for (int i = 0; i <= 255; ++i) {
        double x = i / 255.0;
        double y = 0.0;
        if (x <= controlPoints_.front().x()) {
            y = controlPoints_.front().y();
        } else if (x >= controlPoints_.back().x()) {
            y = controlPoints_.back().y();
        } else {
            for (size_t j = 1; j < controlPoints_.size(); ++j) {
                const QPointF& a = controlPoints_[j - 1];
                const QPointF& b = controlPoints_[j];
                if (x > b.x()) continue;
                double t = (x - a.x()) / std::max(1e-9, b.x() - a.x());
                y = a.y() + (b.y() - a.y()) * t;
                break;
            }
        }
        QPoint pt = curveToWidget(QPointF(x, y));
        if (i == 0) path.moveTo(pt);
        else path.lineTo(pt);
    }

    p.setRenderHint(QPainter::Antialiasing, true);
    p.setPen(QPen(QColor(255, 180, 90), 2.0));
    p.drawPath(path);

    for (size_t i = 0; i < controlPoints_.size(); ++i) {
        QPoint pt = curveToWidget(controlPoints_[i]);
        QColor fill = (static_cast<int>(i) == dragIndex_) ? QColor(255, 220, 120) : QColor(100, 180, 255);
        p.setBrush(fill);
        p.setPen(QColor(20, 20, 20));
        p.drawEllipse(pt, 5, 5);
    }
}

void ToneCurveCanvas::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        dragIndex_ = hitPointIndex(event->pos());
        if (dragIndex_ < 0) {
            QPointF point = widgetToCurve(event->pos());
            controlPoints_.push_back(point);
            clampAndSortControlPoints();
            dragIndex_ = hitPointIndex(curveToWidget(point));
            emit controlPointsChanged(controlPoints_);
            update();
        }
        event->accept();
        return;
    }
    if (event->button() == Qt::RightButton) {
        int index = hitPointIndex(event->pos());
        if (index > 0 && index + 1 < static_cast<int>(controlPoints_.size())) {
            controlPoints_.erase(controlPoints_.begin() + index);
            emit controlPointsChanged(controlPoints_);
            update();
        }
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void ToneCurveCanvas::mouseMoveEvent(QMouseEvent* event)
{
    if (dragIndex_ < 0) {
        QWidget::mouseMoveEvent(event);
        return;
    }

    QPointF point = widgetToCurve(event->pos());
    if (dragIndex_ == 0) {
        point.setX(0.0);
    } else if (dragIndex_ + 1 == static_cast<int>(controlPoints_.size())) {
        point.setX(1.0);
    } else {
        double left = controlPoints_[dragIndex_ - 1].x() + 0.01;
        double right = controlPoints_[dragIndex_ + 1].x() - 0.01;
        point.setX(std::clamp(point.x(), left, right));
    }

    controlPoints_[dragIndex_] = point;
    clampAndSortControlPoints();
    emit controlPointsChanged(controlPoints_);
    update();
}

void ToneCurveCanvas::mouseReleaseEvent(QMouseEvent* event)
{
    dragIndex_ = -1;
    update();
    QWidget::mouseReleaseEvent(event);
}

void ToneCurveCanvas::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        controlPoints_ = {
            QPointF(0.0, 0.0),
            QPointF(0.25, 0.18),
            QPointF(0.5, 0.5),
            QPointF(0.75, 0.82),
            QPointF(1.0, 1.0),
        };
        emit controlPointsChanged(controlPoints_);
        update();
        event->accept();
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

QRect ToneCurveCanvas::graphRect() const
{
    return rect().adjusted(24, 16, -16, -24);
}

QPointF ToneCurveCanvas::widgetToCurve(const QPoint& point) const
{
    QRect graph = graphRect();
    double x = graph.width() > 1 ? (point.x() - graph.left()) / static_cast<double>(graph.width() - 1) : 0.0;
    double y = graph.height() > 1 ? 1.0 - (point.y() - graph.top()) / static_cast<double>(graph.height() - 1) : 0.0;
    return QPointF(std::clamp(x, 0.0, 1.0), std::clamp(y, 0.0, 1.0));
}

QPoint ToneCurveCanvas::curveToWidget(const QPointF& point) const
{
    QRect graph = graphRect();
    int x = graph.left() + static_cast<int>(std::lround(point.x() * (graph.width() - 1)));
    int y = graph.bottom() - static_cast<int>(std::lround(point.y() * (graph.height() - 1)));
    return QPoint(x, y);
}

int ToneCurveCanvas::hitPointIndex(const QPoint& pos) const
{
    for (size_t i = 0; i < controlPoints_.size(); ++i) {
        QPoint pt = curveToWidget(controlPoints_[i]);
        if ((pt - pos).manhattanLength() <= 10) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void ToneCurveCanvas::clampAndSortControlPoints()
{
    std::sort(controlPoints_.begin(), controlPoints_.end(), [](const QPointF& a, const QPointF& b) {
        return a.x() < b.x();
    });
    for (auto& point : controlPoints_) {
        point.setX(std::clamp(point.x(), 0.0, 1.0));
        point.setY(std::clamp(point.y(), 0.0, 1.0));
    }
    controlPoints_.front().setX(0.0);
    controlPoints_.back().setX(1.0);
}

ToneCurveEditorDialog::ToneCurveEditorDialog(QWidget* parent) : QDialog(parent)
{
    setWindowTitle("Tone Curves");
    resize(420, 420);
    auto* layout = new QVBoxLayout(this);
    canvas_ = new ToneCurveCanvas(this);
    layout->addWidget(canvas_);
    connect(canvas_, &ToneCurveCanvas::controlPointsChanged, this, &ToneCurveEditorDialog::curveChanged);
}

void ToneCurveEditorDialog::setControlPoints(const std::vector<QPointF>& points)
{
    canvas_->setControlPoints(points);
}

const std::vector<QPointF>& ToneCurveEditorDialog::controlPoints() const
{
    return canvas_->controlPoints();
}
