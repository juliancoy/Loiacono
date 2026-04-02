#pragma once

#include <QDialog>
#include <QWidget>
#include <QPointF>
#include <vector>

class ToneCurveCanvas : public QWidget {
    Q_OBJECT
public:
    explicit ToneCurveCanvas(QWidget* parent = nullptr);

    void setControlPoints(const std::vector<QPointF>& points);
    const std::vector<QPointF>& controlPoints() const { return controlPoints_; }

signals:
    void controlPointsChanged(const std::vector<QPointF>& points);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    QRect graphRect() const;
    QPointF widgetToCurve(const QPoint& point) const;
    QPoint curveToWidget(const QPointF& point) const;
    int hitPointIndex(const QPoint& pos) const;
    void clampAndSortControlPoints();

    std::vector<QPointF> controlPoints_;
    int dragIndex_ = -1;
};

class ToneCurveEditorDialog : public QDialog {
    Q_OBJECT
public:
    explicit ToneCurveEditorDialog(QWidget* parent = nullptr);

    void setControlPoints(const std::vector<QPointF>& points);
    const std::vector<QPointF>& controlPoints() const;

signals:
    void curveChanged(const std::vector<QPointF>& points);

private:
    ToneCurveCanvas* canvas_ = nullptr;
};
