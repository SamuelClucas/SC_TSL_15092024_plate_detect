Filename: canvas.py
From:
p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)

To:
p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))

From: 
p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())

To:
p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), 
int(self.pixmap.height()))

From: 
p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())

To:
p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), 
int(self.prev_point.y()))

Filename: labelImg.py
From:
bar.setValue(bar.value() + bar.singleStep() * units)

To:
bar.setValue(int(bar.value() + bar.singleStep() * units))

Credit:https://github.com/HumanSignal/labelImg/issues/872#issuecomment-1402362685
