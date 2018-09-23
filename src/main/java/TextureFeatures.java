public enum TextureFeatures {

    ANGULAR_SECOND_MOMENT("Angular Second Moment"),
    CONTRAST("Contrast"),
    CORRELATION("Correlation"),
    SUM_OF_SQUARES("Sum of Squares: Variance"),
    INVERSE_DIFFERENT_MOMENT("Inverse Different Moment"),
    SUM_AVERAGE("Sum Average"),
    SUM_VARIANCE("Sum Variance"),
    SUM_ENTROPY("Sum Entropy"),
    ENTROPY("Entropy"),
    DIFFERENCE_VARIANCE("Difference Variance"),
    DIFFERENCE_ENTROPY("Difference Entropy"),
    F12("Information Measure of Correlation: f12"),
    F13("Information Measure of Correlation: f13"),
    MAXIMAL_CORRELATION_COEFFICIENT("Maximal Correlation Coefficient"),
    INVERSE_MOMENT("Inverse Moment T14"),
    MOMENT("Moment T15"),
    NON_HOMOGENEITY_OF_BRIGHTNESS("Non-homogeneity of Brightness"),
    NON_HOMOGENEITY_OF_SERIES_LENGTH("Non-homogeneity of Series Length"),
    SHARE_OF_IMAGE_IN_SERIES("Share of Image in Series");

    private String description;

    private TextureFeatures( String description ) {
        this.description= description;
    }

    public String getDescription() {
        return description;
    }
};
