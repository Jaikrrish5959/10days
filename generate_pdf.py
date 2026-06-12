import sys
from fpdf import FPDF

class ProjectReportPDF(FPDF):
    def header(self):
        # Set primary color scheme (Navy blue header text)
        self.set_text_color(30, 58, 138)
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Product Sales Demand Forecasting Project', 0, 1, 'C')
        
        self.set_text_color(75, 85, 99)
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 5, 'Comprehensive Project Submission Report', 0, 1, 'C')
        
        # Display Live App URL
        self.set_text_color(30, 58, 138)
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 5, 'Live Web App: 10daysprediction.streamlit.app', 0, 1, 'C')
        
        # Bottom rule line (adjusted down to y=33 to prevent text overlap)
        self.set_draw_color(209, 213, 219)
        self.line(10, 33, 200, 33)
        self.ln(10)

    def footer(self):
        # Position footer at 1.5 cm from bottom
        self.set_y(-15)
        self.set_text_color(156, 163, 175)
        self.set_font('Helvetica', 'I', 8)
        
        # Horizontal rule line
        self.line(10, 280, 200, 280)
        
        # Left-aligned author info, right-aligned page numbers
        self.cell(95, 10, 'Author: Jaikrish | GitHub: Jaikrrish5959', 0, 0, 'L')
        self.cell(95, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'R')

    def section_title(self, label):
        # Section headers styling
        self.set_text_color(30, 58, 138)
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, label, 0, 1, 'L')
        self.ln(1)

    def paragraph(self, text):
        # Standard paragraph text styling
        self.set_text_color(55, 65, 81)
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5.5, text)
        self.ln(4)

    def bullet_point(self, title, description):
        # Bullet point lists formatting
        self.set_text_color(30, 58, 138)
        self.set_font('Helvetica', 'B', 10)
        self.write(5.5, f"  * {title}: ")
        
        self.set_text_color(55, 65, 81)
        self.set_font('Helvetica', '', 10)
        self.write(5.5, f"{description}\n")
        self.ln(2)

def generate_pdf():
    pdf = ProjectReportPDF()
    pdf.alias_nb_pages()
    
    # Page 1: Overview and Preprocessing
    pdf.add_page()
    
    pdf.section_title("1. Project Overview")
    pdf.paragraph(
        "The objective of this project is to build an end-to-end Python system that forecasts daily "
        "sales quantities for individual products (items). The application accepts transaction "
        "data spreadsheets (.xlsx or .csv), cleans them, trains individual forecasting models "
        "for each product, and displays predictions inside a beautiful web-based dashboard."
    )
    
    pdf.section_title("2. The Challenge of Sales Forecasting")
    pdf.paragraph(
        "Predicting consumer demand is challenging because of several real-world data patterns:"
    )
    pdf.bullet_point("Missing Sales Days (Gaps)", "If a product doesn't sell on a specific day, it has no transaction entry. However, to learn cycles (like low weekend sales), the model must know sales were 0. Skipping these days leads to overstated forecasts.")
    pdf.bullet_point("Returned Goods (Negative quantities)", "Returns appear as negative values. These must be filtered out so the model only focuses on outbound purchase demand.")
    pdf.bullet_point("Seasonal Cycles", "Sales are influenced by weekly cycles (which weekdays sell more) and yearly cycles (seasonal holiday buying peaks).")
    pdf.bullet_point("Holidays", "National holiday periods cause sudden peaks or dips in customer buying behavior.")

    pdf.section_title("3. Data Preprocessing & Cleaning Logic")
    pdf.paragraph(
        "Before feeding data into the forecasting models, the pipeline carries out several cleaning steps:"
    )
    pdf.bullet_point("Column Verification", "Verifies the file has required columns (Invoice Date, Product ID, and Quantity).")
    pdf.bullet_point("Filtering Returns", "Excludes all entries with negative quantities since we are forecasting positive demand.")
    pdf.bullet_point("Active Range Trimming", "Trims dates to start from October 1, 2024, to cut out the early system setup/ramp-up period.")
    pdf.bullet_point("Daily Grid Alignment (Zero-Filling)", "Creates a complete sequence of dates. Any date that is missing for a product is filled with 0 sales quantity, ensuring a continuous timeline.")

    # Page 2: Modeling and App features
    pdf.add_page()
    
    pdf.section_title("4. The Forecasting Algorithm (Facebook Prophet)")
    pdf.paragraph(
        "To predict the sales curves, we use Facebook Prophet. Unlike models that predict by "
        "shifting step-by-step from yesterday, Prophet acts like a curve-fitter over time. "
        "It breaks the sales curve into three understandable parts:"
    )
    pdf.bullet_point("Long-Term Trend", "Identifies the overall growth or decline of the product. It automatically flags 'changepoints' where sales growth rates change direction.")
    pdf.bullet_point("Weekly Cycle", "Fits a wave pattern over the 7 days of the week, learning which days are naturally high or low sales days (like low weekend sales).")
    pdf.bullet_point("Yearly Cycle", "Fits an annual wave pattern to capture seasonal trends (like holiday spikes or festive surges).")
    pdf.bullet_point("Holiday Impacts", "Integrates Indian Public Holidays. The model learns how national holidays mathematically shift product demand on those dates.")
    pdf.bullet_point("Multiplicative Mode", "Configures seasonal spikes to multiply (scale) as the overall trend grows, which matches realistic retail behavior (as a store grows, its holiday sales peaks grow too).")

    pdf.section_title("5. Concurrency (Parallel Processing)")
    pdf.paragraph(
        "Fitting individual forecasting models for 200+ distinct products one-by-one is slow and causes "
        "web pages to freeze. We solved this by using a Python Thread Pool. Because Prophet's core solver "
        "is written in compiled C++, it releases Python's lock, allowing the system to train multiple "
        "product models at the same time across your CPU cores. This reduces runtime from minutes to seconds."
    )

    pdf.section_title("6. Web Dashboard Features")
    pdf.paragraph(
        "We built an interactive interface using Streamlit, featuring:"
    )
    pdf.bullet_point("Flexible Forecast Duration Selector", "Allows users to choose to predict in Days (1 to 10 days) or Weeks (1 to 5 weeks).")
    pdf.bullet_point("Weekly Aggregations", "If Weeks are selected, the app runs the forecast daily, but aggregates (sums) the dates into 'Week 1', 'Week 2' columns for clean readability.")
    pdf.bullet_point("Dynamic Highlight Colors", "Color-codes forecast cells (Green = high volume, Orange = medium, Red = low/zero) based on threshold values.")
    pdf.bullet_point("Download Reports", "Exports formatted Excel sheets matching the colors, with frozen panes and auto-fit columns.")
    pdf.bullet_point("Product Inspector", "Renders history vs. forecast plots, confidence bands, and the isolated trend/seasonal sub-curves.")

    # Page 3: Evaluation and Conclusion
    pdf.add_page()
    
    pdf.section_title("7. Model Evaluation & Accuracy Results")
    pdf.paragraph(
        "To evaluate performance, we split the data: training on dates up to Feb 28, 2026, and testing "
        "predictions on March 1 to May 31, 2026. We calculated three standard metrics:"
    )
    pdf.bullet_point("RMSE (Root Mean Squared Error)", "Measures the average unit distance between predicted and actual sales (RMSE on clean data = 1,145 units).")
    pdf.bullet_point("WAPE (Weighted Absolute Percentage Error)", "The retail industry standard for percentage error (avoiding division-by-zero on empty days).")
    pdf.bullet_point("Clean Dataset Results", "On clean product data (IDEAL SALES DATA.xlsx), we achieved a WAPE of 18.95% (under the 20% target, showing an excellent fit).")
    pdf.bullet_point("Noisy Dataset Results", "On noisy data (SAMPLE SALES DATA.xlsx), the WAPE was 70.42% due to heavy return anomalies, showing the value of our data cleaning logic.")

    pdf.section_title("8. Technology Stack Summary")
    pdf.paragraph(
        "The project is built using:"
    )
    pdf.bullet_point("Python 3.13", "Programming language for pipeline logic.")
    pdf.bullet_point("Facebook Prophet", "Time-series forecasting algorithm.")
    pdf.bullet_point("Streamlit", "Dashboard UI framework.")
    pdf.bullet_point("Pandas & NumPy", "Data cleaning and vector manipulation.")
    pdf.bullet_point("Matplotlib", "Visualizations and curve plotting.")
    pdf.bullet_point("Openpyxl", "Excel report generation.")
    pdf.bullet_point("Scikit-Learn", "Regression accuracy calculations.")

    pdf.section_title("9. Conclusion")
    pdf.paragraph(
        "This project successfully builds a robust, scalable product sales demand forecasting system. "
        "By merging high-performance parallel processing, data grid cleaning, Prophet curve fitting, "
        "and user-friendly Streamlit dashboards, it provides supply chain planners with a powerful tool "
        "to manage stock, prevent stock-outs, and optimize operations."
    )
    
    # Save PDF
    pdf.output("Project_Forecasting_Report.pdf")
    print("PDF Report generated successfully as 'Project_Forecasting_Report.pdf'")

if __name__ == "__main__":
    generate_pdf()
